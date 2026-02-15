"""ルックバック日数1~10日を順に増やしてモデル精度の変化を検証する"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from src.fetcher import fetch_stock_list
from src.preprocessor import compute_market_features, compute_pca_factors, get_feature_columns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# --- データ読み込み（1回だけ） ---
print("データ読み込み中...")
stock_list = fetch_stock_list()
prices = pd.read_parquet("data/raw/prices.parquet")
index_data = pd.read_parquet("data/raw/index_prices.parquet")

# 市場指数・PCA・セクター・曜日月は共通なので先に作る
print("共通特徴量を準備中...")
market_feat = compute_market_features(index_data)
if not market_feat.empty:
    for col in market_feat.columns:
        if col != "date":
            market_feat[col] = market_feat[col].shift(1)
    market_feat = market_feat.dropna()

pca_factors, _ = compute_pca_factors(prices, fit=True)

sector_map = stock_list.set_index("symbol")[["sector_code", "sector_name"]].to_dict("index")
le = LabelEncoder()
all_sector_codes = [sector_map.get(s, {}).get("sector_code", "0") for s in prices["symbol"].unique()]
le.fit(all_sector_codes + ["0"])


def build_base_features(prices_df):
    """ベース特徴量を1回だけ構築する（ルックバック非依存部分）。"""
    df = prices_df.sort_values(["symbol", "date"]).copy()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    g = df.groupby("symbol")

    # 出来高（ルックバック非依存）
    if "volume" in df.columns:
        df["volume_change"] = g["volume"].pct_change().shift(1)
        vol_shifted = g["volume"].shift(1)
        vol_ma5 = vol_shifted.groupby(
            df["symbol"]
        ).rolling(5, min_periods=1).mean().droplevel(0)
        df["volume_ma5_ratio"] = df["volume"] / vol_ma5

    # ターゲット
    df["next_open"] = g["open"].shift(-1)
    df["next_close"] = g["close"].shift(-1)
    df["target"] = (df["next_close"] > df["next_open"]).astype(int)

    # 市場指数
    if not market_feat.empty:
        df = df.merge(market_feat, on="date", how="left")

    # PCA
    if not pca_factors.empty:
        df = df.merge(pca_factors, on="date", how="left")

    # セクター
    sector_code_map = {s: v.get("sector_code", "0") for s, v in sector_map.items()}
    df["sector_code"] = df["symbol"].map(sector_code_map).fillna("0")
    df["sector_encoded"] = le.transform(df["sector_code"].astype(str))

    # 曜日・月
    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    # 前日シフト（rolling用に事前計算）
    intraday_shifted = g["intraday_return"].shift(1)
    daily_shifted = g["daily_return"].shift(1)
    close_shifted = g["close"].shift(1)

    return df, intraday_shifted, daily_shifted, close_shifted


def add_lookback_features(base_df, intraday_shifted, daily_shifted, close_shifted, n,
                          intraday_grp=None, daily_grp=None, close_grp=None):
    """ルックバックn日の3列を追加する（vectorized）。"""
    if intraday_grp is None:
        sym = base_df["symbol"]
        intraday_grp = intraday_shifted.groupby(sym)
        daily_grp = daily_shifted.groupby(sym)
        close_grp = close_shifted.groupby(sym)
    base_df[f"intraday_ret_ma{n}"] = intraday_grp.rolling(n, min_periods=1).mean().droplevel(0)
    base_df[f"volatility_{n}d"] = daily_grp.rolling(n, min_periods=1).std().droplevel(0)
    close_ma = close_grp.rolling(n, min_periods=1).mean().droplevel(0)
    base_df[f"ma{n}_deviation"] = base_df["close"] / close_ma - 1


def evaluate_model(dataset):
    """モデルを学習してテスト期間のTOP-N正解率を返す。"""
    feature_cols = get_feature_columns(dataset)
    df = dataset.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_end = dates[int(n * 0.7)]
    valid_end = dates[int(n * 0.85)]

    train = df[df["date"] <= train_end]
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)]
    test = df[df["date"] > valid_end]

    X_train, y_train = train[feature_cols], train["target"]
    X_valid, y_valid = valid[feature_cols], valid["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.log_evaluation(period=999), lgb.early_stopping(stopping_rounds=30)],
        categorical_feature=[c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")] or "auto",
    )

    # テスト期間全体の評価
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, test_pred)
    auc = roc_auc_score(y_test, test_proba)

    # TOP-N正解率
    test = test.copy()
    test["proba"] = test_proba
    test_dates = np.sort(test["date"].unique())
    baseline = test["target"].mean()

    results = {"acc": acc, "auc": auc, "baseline": baseline, "n_features": len(feature_cols)}
    # 日次データを事前グループ化してO(1)参照
    test_date_groups = {d: g for d, g in test.groupby("date")}
    for top_n in [1, 5, 10, 20]:
        hits = 0
        total = 0
        for d in test_dates:
            if d not in test_date_groups:
                continue
            top = test_date_groups[d].nlargest(top_n, "proba")
            hits += int(top["target"].sum())
            total += len(top)
        results[f"top{top_n}"] = hits / total if total > 0 else 0
    return results


# --- メインループ: ルックバック1日~10日 ---
print()
print("=" * 85)
print("  ルックバック日数別 モデル精度検証")
print("=" * 85)
print()

all_results = []

# ベース特徴量を1回だけ構築
print("ベース特徴量を構築中...")
base_df, intraday_shifted, daily_shifted, close_shifted = build_base_features(prices)

# groupbyオブジェクトを事前作成して全ルックバック日で使い回す
sym = base_df["symbol"]
intraday_grp = intraday_shifted.groupby(sym)
daily_grp = daily_shifted.groupby(sym)
close_grp = close_shifted.groupby(sym)

for max_day in range(1, 11):
    print(f"--- ルックバック {max_day}日 (特徴量窓: {list(range(1, max_day + 1))}) ---")

    # 新しいルックバック日のみ3列追加（事前作成したgroupbyオブジェクトを再利用）
    add_lookback_features(base_df, intraday_shifted, daily_shifted, close_shifted, max_day,
                          intraday_grp=intraday_grp, daily_grp=daily_grp, close_grp=close_grp)

    res = evaluate_model(base_df)
    res["lookback"] = max_day
    all_results.append(res)

    print(f"  特徴量数={res['n_features']}, AUC={res['auc']:.4f}, "
          f"TOP1={res['top1']*100:.1f}%, TOP5={res['top5']*100:.1f}%, "
          f"TOP10={res['top10']*100:.1f}%, TOP20={res['top20']*100:.1f}%")
    print()

# --- 結果サマリ ---
print()
print("=" * 85)
print(f"{'日数':>4}  {'特徴量数':>6}  {'AUC':>6}  {'Acc':>6}  {'TOP1':>6}  {'TOP5':>6}  {'TOP10':>6}  {'TOP20':>6}  {'ベースライン':>8}")
print("-" * 85)
for r in all_results:
    print(f"{r['lookback']:>4}  {r['n_features']:>6}  {r['auc']:.4f}  {r['acc']*100:.1f}%  "
          f"{r['top1']*100:.1f}%  {r['top5']*100:.1f}%  {r['top10']*100:.1f}%  "
          f"{r['top20']*100:.1f}%  {r['baseline']*100:.1f}%")
print("-" * 85)
print("※ ベースライン = テスト期間全銘柄の平均上昇率")
