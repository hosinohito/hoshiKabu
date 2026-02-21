"""ウォークフォワード検証（新戦略）。

翌日始値基準で以下2イベントを別々に評価する:
- HIGH_5PCT: 翌日高値が始値より5%以上高い
- LOW_5PCT: 翌日安値が始値より5%以上低い
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from src.fetcher import fetch_stock_list
from src.preprocessor import build_dataset, get_feature_columns

VOLUME_THRESHOLD = config.VOLUME_THRESHOLD
RETRAIN_INTERVAL = 10

FAST_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "is_unbalance": True,
    "min_child_samples": 20,
    "n_estimators": 400,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}


def train_one_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_date,
    cat_features: list[str],
    scale_pos_weight: float,
):
    train_data = df[df["date"] < test_date]
    if len(train_data) < 1000:
        return None

    train_dates_all = np.sort(train_data["date"].unique())
    valid_start = train_dates_all[int(len(train_dates_all) * 0.9)]
    train_part = train_data[train_data["date"] < valid_start]
    valid_part = train_data[train_data["date"] >= valid_start]
    if valid_part.empty:
        valid_part = train_part.tail(max(1, len(train_part) // 5))

    X_train = train_part[feature_cols].fillna(0)
    y_train = train_part[target_col]
    X_valid = valid_part[feature_cols].fillna(0)
    y_valid = valid_part[target_col]

    params = {**FAST_PARAMS, "scale_pos_weight": scale_pos_weight, "is_unbalance": False}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.log_evaluation(period=50)],
        categorical_feature=cat_features or "auto",
    )
    return model


def evaluate_day(day_df: pd.DataFrame, proba: np.ndarray, target_col: str):
    work = day_df.copy()
    work["proba"] = proba
    if "volume" in work.columns:
        work = work[work["volume"] >= VOLUME_THRESHOLD]
    if work.empty:
        return None

    top1 = work.nlargest(1, "proba").iloc[0]
    topn_acc = {}
    for topn in [1, 3, 5]:
        topn_df = work.nlargest(topn, "proba")
        topn_acc[topn] = float((topn_df[target_col] == 1).mean())

    return {
        "symbol": top1["symbol"],
        "confidence": float(top1["proba"]),
        "correct": int(top1[target_col] == 1),
        "top1_acc": topn_acc[1],
        "top3_acc": topn_acc[3],
        "top5_acc": topn_acc[5],
    }


print("データ読み込み中...")
stock_list = fetch_stock_list()
prices = pd.read_parquet("data/raw/prices.parquet")
index_data = pd.read_parquet("data/raw/index_prices.parquet")

print("データセット構築中...")
# 評価時は直近3年に限定
config.TRAIN_LOOKBACK_YEARS = 3
dataset = build_dataset(prices, stock_list, index_data, pca_fit=True)
feature_cols = get_feature_columns(dataset)
cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]

targets = ["target_high_5pct", "target_low_5pct"]
df = dataset.dropna(subset=targets).copy()
for col in targets:
    df[col] = df[col].astype(int)

# ターゲット別の不均衡率を算出し、scale_pos_weightに反映
target_weights: dict[str, float] = {}
for col in targets:
    pos = int(df[col].sum())
    neg = int(len(df) - pos)
    weight = (neg / max(pos, 1))
    target_weights[col] = weight
    print(f"{col}: pos={pos:,}, neg={neg:,}, pos_rate={pos/len(df)*100:.2f}%, scale_pos_weight={weight:.2f}")

dates = np.sort(df["date"].unique())
n = len(dates)
test_start_idx = int(n * 0.90)
test_dates = dates[test_start_idx:]

print(f"テスト期間: {pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()} ({len(test_dates)}日)")
print(f"リトレイン間隔: {RETRAIN_INTERVAL}営業日ごと")
print(f"出来高フィルタ: >= {VOLUME_THRESHOLD:,}")
print()

date_groups = {date: group for date, group in df.groupby("date")}
models = {t: None for t in targets}
results = {t: [] for t in targets}

for i, test_date in enumerate(test_dates):
    if models[targets[0]] is None or i % RETRAIN_INTERVAL == 0:
        for target_col in targets:
            models[target_col] = train_one_model(
                df,
                feature_cols,
                target_col,
                test_date,
                cat_features,
                target_weights[target_col],
            )

    if test_date not in date_groups:
        continue
    day_df = date_groups[test_date]

    for target_col in targets:
        model = models[target_col]
        if model is None:
            continue
        proba = model.predict_proba(day_df[feature_cols].fillna(0))[:, 1]
        day_result = evaluate_day(day_df, proba, target_col)
        if day_result is None:
            continue
        results[target_col].append({"date": test_date, **day_result, "retrained": (i % RETRAIN_INTERVAL == 0)})

    if (i + 1) % 20 == 0:
        def _acc(t):
            return (sum(r["correct"] for r in results[t]) / len(results[t]) * 100) if results[t] else 0.0
        print(
            f"  進捗: {i+1}/{len(test_dates)}日完了, "
            f"HIGH+5%={_acc('target_high_5pct'):.1f}% LOW-5%={_acc('target_low_5pct'):.1f}%"
        )

print()
print("=" * 85)
print("  ウォークフォワード検証結果（新戦略）")
print("=" * 85)
print()

for target_col, title in [
    ("target_high_5pct", "HIGH_5PCT: 翌日高値が始値より5%以上高い"),
    ("target_low_5pct", "LOW_5PCT: 翌日安値が始値より5%以上低い"),
]:
    rdf = pd.DataFrame(results[target_col])
    if rdf.empty:
        print(f"【{title}】データなし")
        print()
        continue

    total = len(rdf)
    correct = int(rdf["correct"].sum())
    print(f"【{title}】")
    print(f"  TOP1 正解率: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  TOP3 正解率: {rdf['top3_acc'].mean()*100:.1f}%")
    print(f"  TOP5 正解率: {rdf['top5_acc'].mean()*100:.1f}%")

    rdf["month"] = pd.to_datetime(rdf["date"]).dt.to_period("M").astype(str)
    print(f"  月別TOP1:")
    print(f"{'月':>10}  {'日数':>4}  {'正解':>4}  {'正解率':>7}")
    print("-" * 36)
    for month, g in rdf.groupby("month"):
        days = len(g)
        hits = int(g["correct"].sum())
        rate = hits / days * 100
        print(f"{month:>10}  {days:>4}  {hits:>4}  {rate:>6.1f}%")
    print("-" * 36)
    print()
