"""ウォークフォワード検証: 1日ずつ学習→予測を繰り返し、データリーク無しで正解率を算出する。

毎日学習すると遅いため、5営業日ごとにリトレインし、間の日は同じモデルで予測する。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

import config
from src.fetcher import fetch_stock_list
from src.preprocessor import build_dataset, get_feature_columns

VOLUME_THRESHOLD = config.VOLUME_THRESHOLD
RETRAIN_INTERVAL = 5  # 5営業日ごとにリトレイン

# 高速化用パラメータ
FAST_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "is_unbalance": True,
    "n_estimators": 150,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}

print("データ読み込み中...")
stock_list = fetch_stock_list()
prices = pd.read_parquet("data/raw/prices.parquet")
index_data = pd.read_parquet("data/raw/index_prices.parquet")

print("データセット構築中...")
dataset = build_dataset(prices, stock_list, index_data, pca_fit=True)
feature_cols = get_feature_columns(dataset)

df = dataset.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

# テスト期間: 後ろ15%
dates = np.sort(df["date"].unique())
n = len(dates)
test_start_idx = int(n * 0.85)
test_dates = dates[test_start_idx:]
# 検証用分割: 学習データの最後10%をvalidationに使う
valid_ratio = 0.1

cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]

print(f"テスト期間: {pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()} ({len(test_dates)}日)")
print(f"リトレイン間隔: {RETRAIN_INTERVAL}営業日ごと")
print(f"出来高フィルタ: >= {VOLUME_THRESHOLD:,}")
print()

name_map = stock_list.set_index("symbol")["name"].to_dict()

model = None
results = []

# 日次データを事前にグループ化してO(1)参照可能にする
date_groups = {date: group for date, group in df.groupby("date")}

for i, test_date in enumerate(test_dates):
    # リトレインするか判定
    if model is None or i % RETRAIN_INTERVAL == 0:
        # test_dateより前のデータのみで学習
        train_data = df[df["date"] < test_date]
        if len(train_data) < 1000:
            continue

        # 学習データの最後10%をvalidationに
        train_dates_all = np.sort(train_data["date"].unique())
        valid_start = train_dates_all[int(len(train_dates_all) * (1 - valid_ratio))]
        train_part = train_data[train_data["date"] < valid_start]
        valid_part = train_data[train_data["date"] >= valid_start]

        model = lgb.LGBMClassifier(**FAST_PARAMS)
        model.fit(
            train_part[feature_cols], train_part["target"],
            eval_set=[(valid_part[feature_cols], valid_part["target"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.log_evaluation(period=999), lgb.early_stopping(stopping_rounds=20)],
            categorical_feature=cat_features or "auto",
        )

    # test_dateの予測（事前グループ化した辞書からO(1)で取得）
    if test_date not in date_groups:
        continue
    day_df = date_groups[test_date].copy()

    # 出来高フィルタ
    if "volume" in day_df.columns:
        day_df = day_df[day_df["volume"] >= VOLUME_THRESHOLD]
    if day_df.empty:
        continue

    proba = model.predict_proba(day_df[feature_cols])[:, 1]
    day_df["proba"] = proba

    # 上昇TOP1（nlargest/nsmallestでO(n)ヒープ選択）
    up_top = day_df.nlargest(1, "proba").iloc[0]
    up_conf = up_top["proba"]

    # 値下がりTOP1
    dn_top = day_df.nsmallest(1, "proba").iloc[0]
    dn_conf = 1 - dn_top["proba"]

    # 統合: 確信度の高い方を採用
    if up_conf >= dn_conf:
        direction = "UP"
        chosen = up_top
        correct = int(chosen["target"] == 1)
        conf = up_conf
    else:
        direction = "DOWN"
        chosen = dn_top
        correct = int(chosen["target"] == 0)
        conf = dn_conf

    # UP/DOWN独立 TOP-N評価用（nlargest/nsmallestで高速化）
    up_topn_correct = {}
    dn_topn_correct = {}
    for topn in [1, 3, 5]:
        up_topn = day_df.nlargest(topn, "proba")
        dn_topn = day_df.nsmallest(topn, "proba")
        up_topn_correct[topn] = (up_topn["target"] == 1).mean()
        dn_topn_correct[topn] = (dn_topn["target"] == 0).mean()

    results.append({
        "date": test_date,
        "direction": direction,
        "symbol": chosen["symbol"],
        "name": name_map.get(chosen["symbol"], ""),
        "confidence": conf,
        "correct": correct,
        "up_conf": up_conf,
        "dn_conf": dn_conf,
        "up_symbol": up_top["symbol"],
        "up_correct": int(up_top["target"] == 1),
        "dn_symbol": dn_top["symbol"],
        "dn_correct": int(dn_top["target"] == 0),
        "retrained": (i % RETRAIN_INTERVAL == 0),
        **{f"up_top{n}_acc": up_topn_correct[n] for n in [1, 3, 5]},
        **{f"dn_top{n}_acc": dn_topn_correct[n] for n in [1, 3, 5]},
    })

    if (i + 1) % 20 == 0:
        done = len(results)
        acc = sum(r["correct"] for r in results) / done * 100
        print(f"  進捗: {i+1}/{len(test_dates)}日完了, 暫定正解率={acc:.1f}%")

rdf = pd.DataFrame(results)

# === 結果表示 ===
print()
print("=" * 85)
print("  ウォークフォワード検証結果（データリーク無し）")
print("=" * 85)
print()

total = len(rdf)
correct = int(rdf["correct"].sum())
up_days = rdf[rdf["direction"] == "UP"]
dn_days = rdf[rdf["direction"] == "DOWN"]

print(f"【統合戦略 TOP1】")
print(f"  全体:       正解 {correct}/{total}  正解率 {correct/total*100:.1f}%")
if len(up_days) > 0:
    print(f"  上昇採用:   正解 {int(up_days['correct'].sum())}/{len(up_days)}  正解率 {up_days['correct'].mean()*100:.1f}%  ({len(up_days)}日)")
else:
    print(f"  上昇採用:   0日")
if len(dn_days) > 0:
    print(f"  値下がり採用: 正解 {int(dn_days['correct'].sum())}/{len(dn_days)}  正解率 {dn_days['correct'].mean()*100:.1f}%  ({len(dn_days)}日)")
else:
    print(f"  値下がり採用: 0日")
print()

print(f"【参考: 単独戦略】")
up_only = int(rdf["up_correct"].sum())
dn_only = int(rdf["dn_correct"].sum())
print(f"  上昇のみ:   正解 {up_only}/{total}  正解率 {up_only/total*100:.1f}%")
print(f"  値下がりのみ: 正解 {dn_only}/{total}  正解率 {dn_only/total*100:.1f}%")
print()

print(f"【UP/DOWN独立 TOP-N正解率】")
print(f"  {'':>12}  {'TOP1':>7}  {'TOP3':>7}  {'TOP5':>7}")
print(f"  {'上昇予測':>12}  {rdf['up_top1_acc'].mean()*100:>6.1f}%  {rdf['up_top3_acc'].mean()*100:>6.1f}%  {rdf['up_top5_acc'].mean()*100:>6.1f}%")
print(f"  {'値下がり予測':>12}  {rdf['dn_top1_acc'].mean()*100:>6.1f}%  {rdf['dn_top3_acc'].mean()*100:>6.1f}%  {rdf['dn_top5_acc'].mean()*100:>6.1f}%")
print()

# 月別
rdf["month"] = pd.to_datetime(rdf["date"]).dt.to_period("M").astype(str)
print(f"【月別 正解率】")
print(f"{'月':>10}  {'日数':>4}  {'正解':>4}  {'正解率':>7}  {'上昇日':>5}  {'下落日':>5}  {'上昇のみ':>7}  {'下落のみ':>7}")
print("-" * 80)
for month, g in rdf.groupby("month"):
    days = len(g)
    hits = int(g["correct"].sum())
    rate = hits / days * 100
    n_up = len(g[g["direction"] == "UP"])
    n_dn = len(g[g["direction"] == "DOWN"])
    up_r = int(g["up_correct"].sum()) / days * 100
    dn_r = int(g["dn_correct"].sum()) / days * 100
    print(f"{month:>10}  {days:>4}  {hits:>4}  {rate:>6.1f}%  {n_up:>5}  {n_dn:>5}  {up_r:>6.1f}%  {dn_r:>6.1f}%")

print("-" * 80)
up_r_total = up_only / total * 100
dn_r_total = dn_only / total * 100
print(f"{'合計':>10}  {total:>4}  {correct:>4}  {correct/total*100:>6.1f}%  {len(up_days):>5}  {len(dn_days):>5}  {up_r_total:>6.1f}%  {dn_r_total:>6.1f}%")
