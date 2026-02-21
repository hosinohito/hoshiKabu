"""ルックバック日数ごとのTOP1正解率比較（新戦略）。

対象ターゲット:
- target_high_5pct: 翌日高値が始値より5%以上高い
- target_low_5pct: 翌日安値が始値より5%以上低い
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

import config
from src.fetcher import fetch_stock_list
from src.preprocessor import compute_market_features, compute_pca_factors, get_feature_columns


FAST_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.02,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "min_child_samples": 100,
    "n_estimators": 2000,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}


def _build_base_features(
    prices_df: pd.DataFrame,
    market_feat: pd.DataFrame,
    pca_factors: pd.DataFrame,
    sector_map: dict,
    le: LabelEncoder,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = prices_df.sort_values(["symbol", "date"]).copy()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    g = df.groupby("symbol")

    # 出来高（ルックバック非依存）
    if "volume" in df.columns:
        df["volume_change"] = g["volume"].pct_change().shift(1)
        vol_shifted = g["volume"].shift(1)
        vol_ma5 = vol_shifted.groupby(df["symbol"]).rolling(5, min_periods=1).mean().droplevel(0)
        df["volume_ma5_ratio"] = df["volume"] / vol_ma5

    # 新ターゲット
    df["next_open"] = g["open"].shift(-1)
    df["next_high"] = g["high"].shift(-1)
    df["next_low"] = g["low"].shift(-1)
    df["target_high_5pct"] = (df["next_high"] >= df["next_open"] * 1.05).astype(int)
    df["target_low_5pct"] = (df["next_low"] <= df["next_open"] * 0.95).astype(int)

    if not market_feat.empty:
        df = df.merge(market_feat, on="date", how="left")
    if not pca_factors.empty:
        df = df.merge(pca_factors, on="date", how="left")

    sector_code_map = {s: v.get("sector_code", "0") for s, v in sector_map.items()}
    df["sector_code"] = df["symbol"].map(sector_code_map).fillna("0")
    df["sector_encoded"] = le.transform(df["sector_code"].astype(str))

    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    # rolling用の事前シフト
    intraday_shifted = g["intraday_return"].shift(1)
    daily_shifted = g["daily_return"].shift(1)
    close_shifted = g["close"].shift(1)
    return df, intraday_shifted, daily_shifted, close_shifted


def _add_lookback_features(
    base_df: pd.DataFrame,
    intraday_shifted: pd.Series,
    daily_shifted: pd.Series,
    close_shifted: pd.Series,
    n: int,
    intraday_grp=None,
    daily_grp=None,
    close_grp=None,
) -> None:
    if intraday_grp is None:
        sym = base_df["symbol"]
        intraday_grp = intraday_shifted.groupby(sym)
        daily_grp = daily_shifted.groupby(sym)
        close_grp = close_shifted.groupby(sym)
    base_df[f"intraday_ret_ma{n}"] = intraday_grp.rolling(n, min_periods=1).mean().droplevel(0)
    base_df[f"volatility_{n}d"] = daily_grp.rolling(n, min_periods=1).std().droplevel(0)
    close_ma = close_grp.rolling(n, min_periods=1).mean().droplevel(0)
    base_df[f"ma{n}_deviation"] = base_df["close"] / close_ma - 1


def _calc_top1_accuracy(df_test: pd.DataFrame, proba: np.ndarray, target_col: str) -> float:
    work = df_test.copy()
    work["proba"] = proba
    if "volume" in work.columns:
        work = work[work["volume"] >= config.VOLUME_THRESHOLD]
    if work.empty:
        return 0.0
    day_groups = {d: g for d, g in work.groupby("date")}
    hits = 0
    days = 0
    for d in sorted(day_groups.keys()):
        top1 = day_groups[d].nlargest(1, "proba")
        if top1.empty:
            continue
        hits += int(top1[target_col].iloc[0] == 1)
        days += 1
    return hits / days if days > 0 else 0.0


def _evaluate_top1(dataset: pd.DataFrame, target_col: str) -> tuple[float, int]:
    feature_cols = get_feature_columns(dataset)
    df = dataset.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_end = dates[int(n * 0.7)]
    valid_end = dates[int(n * 0.85)]

    train = df[df["date"] <= train_end]
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)]
    test = df[df["date"] > valid_end]

    X_train = train[feature_cols].fillna(0)
    y_train = train[target_col]
    X_valid = valid[feature_cols].fillna(0)
    y_valid = valid[target_col]
    X_test = test[feature_cols].fillna(0)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    raw_ratio = neg / max(pos, 1)
    scale_pos_weight = min(10.0, np.sqrt(raw_ratio))

    params = {**FAST_PARAMS, "scale_pos_weight": scale_pos_weight, "is_unbalance": False}
    model = lgb.LGBMClassifier(**params)
    cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")] or "auto"
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.log_evaluation(period=200), lgb.early_stopping(stopping_rounds=150)],
        categorical_feature=cat_features,
    )

    proba = model.predict_proba(X_test)[:, 1]
    return _calc_top1_accuracy(test, proba, target_col), len(feature_cols)


def main() -> None:
    print("データ読み込み中...")
    stock_list = fetch_stock_list()
    prices = pd.read_parquet("data/raw/prices.parquet")
    index_data = pd.read_parquet("data/raw/index_prices.parquet")

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

    print("ベース特徴量を構築中...")
    base_df, intraday_shifted, daily_shifted, close_shifted = _build_base_features(
        prices, market_feat, pca_factors, sector_map, le
    )

    sym = base_df["symbol"]
    intraday_grp = intraday_shifted.groupby(sym)
    daily_grp = daily_shifted.groupby(sym)
    close_grp = close_shifted.groupby(sym)

    rows = []
    print()
    print("=" * 88)
    print("  ルックバック日数別 TOP1 正解率比較（新戦略）")
    print("=" * 88)
    print()

    for max_day in range(1, 11):
        _add_lookback_features(
            base_df,
            intraday_shifted,
            daily_shifted,
            close_shifted,
            max_day,
            intraday_grp=intraday_grp,
            daily_grp=daily_grp,
            close_grp=close_grp,
        )
        high_top1, n_features = _evaluate_top1(base_df, "target_high_5pct")
        low_top1, _ = _evaluate_top1(base_df, "target_low_5pct")
        avg_top1 = (high_top1 + low_top1) / 2
        rows.append(
            {
                "lookback_max_day": max_day,
                "n_features": n_features,
                "high_top1": high_top1,
                "low_top1": low_top1,
                "avg_top1": avg_top1,
            }
        )
        print(
            f"lookback=1..{max_day:<2}  "
            f"features={n_features:<3}  "
            f"HIGH_TOP1={high_top1*100:5.1f}%  "
            f"LOW_TOP1={low_top1*100:5.1f}%  "
            f"AVG={avg_top1*100:5.1f}%"
        )

    res = pd.DataFrame(rows).sort_values("lookback_max_day")
    print()
    print("-" * 88)
    print(res.to_string(index=False, formatters={
        "high_top1": lambda x: f"{x*100:.1f}%",
        "low_top1": lambda x: f"{x*100:.1f}%",
        "avg_top1": lambda x: f"{x*100:.1f}%",
    }))
    print("-" * 88)

    best_high = res.iloc[res["high_top1"].idxmax()]
    best_low = res.iloc[res["low_top1"].idxmax()]
    best_avg = res.iloc[res["avg_top1"].idxmax()]
    print(
        f"BEST_HIGH: lookback=1..{int(best_high['lookback_max_day'])}, "
        f"TOP1={best_high['high_top1']*100:.1f}%"
    )
    print(
        f"BEST_LOW : lookback=1..{int(best_low['lookback_max_day'])}, "
        f"TOP1={best_low['low_top1']*100:.1f}%"
    )
    print(
        f"BEST_AVG : lookback=1..{int(best_avg['lookback_max_day'])}, "
        f"TOP1={best_avg['avg_top1']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
