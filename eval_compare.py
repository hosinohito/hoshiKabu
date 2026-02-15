"""Baseline vs Enhanced ウォークフォワード比較検証。

同一ループ内で Baseline (旧特徴量・単一モデル) と Enhanced (新特徴量・アンサンブル・
キャリブレーション・非対称閾値・サンプル重み) を同時実行し公平に比較する。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

import config
from src.fetcher import fetch_stock_list
from src.preprocessor import build_dataset, get_feature_columns

VOLUME_THRESHOLD = config.VOLUME_THRESHOLD
RETRAIN_INTERVAL = 5

# Baseline: eval_walkforward.py と同一パラメータ
BASELINE_PARAMS = {
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

# Enhanced: 軽量アンサンブル (eval用)
ENSEMBLE_COMMON = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "bagging_freq": 5,
    "verbose": -1,
    "is_unbalance": True,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}

ENSEMBLE_FAST = [
    {  # A: 標準
        "num_leaves": 63, "learning_rate": 0.1, "n_estimators": 150,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "min_child_samples": 50,
    },
    {  # B: 浅く速く
        "num_leaves": 31, "learning_rate": 0.15, "n_estimators": 120,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "min_child_samples": 100,
    },
    {  # C: 深く正則化
        "num_leaves": 127, "learning_rate": 0.08, "n_estimators": 180,
        "feature_fraction": 0.6, "bagging_fraction": 0.9, "min_child_samples": 30,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
]

# 新特徴量セット (Baselineから除外する)
NEW_FEATURES = {
    "rsi", "macd_histogram", "macd_normalized",
    "bollinger_position", "bollinger_width",
    "volume_spike_ratio", "sector_relative_strength",
}


def train_baseline(train_part, valid_part, feature_cols, cat_features):
    """Baseline: 単一LightGBMモデルを学習。"""
    model = lgb.LGBMClassifier(**BASELINE_PARAMS)
    model.fit(
        train_part[feature_cols], train_part["target"],
        eval_set=[(valid_part[feature_cols], valid_part["target"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.log_evaluation(period=999), lgb.early_stopping(stopping_rounds=20)],
        categorical_feature=cat_features or "auto",
    )
    return model


def train_enhanced(train_part, calib_part, valid_part, feature_cols, cat_features):
    """Enhanced: アンサンブル3モデル + キャリブレーション。"""
    # サンプル重み
    weights = _compute_weights(train_part["date"].values, config.SAMPLE_WEIGHT_DECAY)

    # 3モデル学習
    models = []
    for params in ENSEMBLE_FAST:
        merged = {**ENSEMBLE_COMMON, **params}
        m = lgb.LGBMClassifier(**merged)
        m.fit(
            train_part[feature_cols], train_part["target"],
            eval_set=[(valid_part[feature_cols], valid_part["target"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.log_evaluation(period=999), lgb.early_stopping(stopping_rounds=20)],
            categorical_feature=cat_features or "auto",
            sample_weight=weights,
        )
        models.append(m)

    # キャリブレーション
    calib_proba = np.column_stack(
        [m.predict_proba(calib_part[feature_cols])[:, 1] for m in models]
    ).mean(axis=1)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(calib_proba, calib_part["target"].values)

    return models, calibrator


def _compute_weights(dates, decay):
    """指数減衰サンプル重み。"""
    unique = np.sort(np.unique(dates))
    d2i = {d: i for i, d in enumerate(unique)}
    max_i = len(unique) - 1
    return decay ** (max_i - np.array([d2i[d] for d in dates]))


def evaluate_day(day_df, proba, direction_func):
    """1日分の予測結果を評価する。direction_funcは(day_df, proba)->結果dictを返す。"""
    day_df = day_df.copy()
    day_df["proba"] = proba

    # 出来高フィルタ
    if "volume" in day_df.columns:
        day_df = day_df[day_df["volume"] >= VOLUME_THRESHOLD]
    if day_df.empty:
        return None

    return direction_func(day_df)


def baseline_direction(day_df):
    """Baseline: 対称閾値（0.5基準、確信度比較）。"""
    up_top = day_df.nlargest(1, "proba").iloc[0]
    dn_top = day_df.nsmallest(1, "proba").iloc[0]
    up_conf = up_top["proba"]
    dn_conf = 1 - dn_top["proba"]

    if up_conf >= dn_conf:
        return {"direction": "UP", "correct": int(up_top["target"] == 1),
                "up_correct": int(up_top["target"] == 1), "dn_correct": int(dn_top["target"] == 0)}
    else:
        return {"direction": "DOWN", "correct": int(dn_top["target"] == 0),
                "up_correct": int(up_top["target"] == 1), "dn_correct": int(dn_top["target"] == 0)}


def enhanced_direction(day_df):
    """Enhanced: 非対称閾値。"""
    up_top = day_df.nlargest(1, "proba").iloc[0]
    dn_top = day_df.nsmallest(1, "proba").iloc[0]
    up_conf = up_top["proba"]
    dn_conf = 1 - dn_top["proba"]

    up_passes = up_conf >= config.UP_THRESHOLD
    dn_passes = dn_conf >= config.DOWN_THRESHOLD

    if up_passes and dn_passes:
        direction = "UP" if up_conf >= dn_conf else "DOWN"
    elif up_passes:
        direction = "UP"
    elif dn_passes:
        direction = "DOWN"
    else:
        direction = "UP" if up_conf >= dn_conf else "DOWN"

    if direction == "UP":
        correct = int(up_top["target"] == 1)
    else:
        correct = int(dn_top["target"] == 0)

    return {"direction": direction, "correct": correct,
            "up_correct": int(up_top["target"] == 1), "dn_correct": int(dn_top["target"] == 0)}


# === メイン実行 ===
print("=" * 85)
print("  Baseline vs Enhanced Walk-Forward Comparison")
print("=" * 85)
print()

print("データ読み込み中...")
stock_list = fetch_stock_list()
prices = pd.read_parquet("data/raw/prices.parquet")
index_data = pd.read_parquet("data/raw/index_prices.parquet")

print("データセット構築中 (新特徴量含む)...")
dataset = build_dataset(prices, stock_list, index_data, pca_fit=True)
all_feature_cols = get_feature_columns(dataset)

# 特徴量の分離
baseline_feature_cols = [c for c in all_feature_cols if c not in NEW_FEATURES]
enhanced_feature_cols = all_feature_cols

print(f"Baseline 特徴量: {len(baseline_feature_cols)}列")
print(f"Enhanced 特徴量: {len(enhanced_feature_cols)}列")
print(f"新規追加特徴量: {sorted(NEW_FEATURES & set(all_feature_cols))}")
print()

df = dataset.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

# テスト期間: 後ろ15%
dates = np.sort(df["date"].unique())
n = len(dates)
test_start_idx = int(n * 0.85)
test_dates = dates[test_start_idx:]

cat_features_base = [c for c in baseline_feature_cols if c in ("sector_encoded", "dayofweek", "month")]
cat_features_enh = [c for c in enhanced_feature_cols if c in ("sector_encoded", "dayofweek", "month")]

print(f"テスト期間: {pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()} ({len(test_dates)}日)")
print(f"リトレイン間隔: {RETRAIN_INTERVAL}営業日ごと")
print()

# 日次データの事前グループ化
date_groups = {date: group for date, group in df.groupby("date")}

baseline_model = None
enhanced_models = None
enhanced_calibrator = None

baseline_results = []
enhanced_results = []

for i, test_date in enumerate(test_dates):
    need_retrain = (baseline_model is None) or (i % RETRAIN_INTERVAL == 0)

    if need_retrain:
        train_data = df[df["date"] < test_date]
        if len(train_data) < 1000:
            continue

        train_dates_all = np.sort(train_data["date"].unique())
        n_train = len(train_dates_all)

        # Baseline: 90% train, 10% valid
        bl_valid_start = train_dates_all[int(n_train * 0.9)]
        bl_train = train_data[train_data["date"] < bl_valid_start]
        bl_valid = train_data[train_data["date"] >= bl_valid_start]

        baseline_model = train_baseline(bl_train, bl_valid, baseline_feature_cols, cat_features_base)

        # Enhanced: 85% train, 5% calib, 10% valid
        enh_calib_start = train_dates_all[int(n_train * 0.85)]
        enh_valid_start = train_dates_all[int(n_train * 0.90)]
        enh_train = train_data[train_data["date"] < enh_calib_start]
        enh_calib = train_data[(train_data["date"] >= enh_calib_start) & (train_data["date"] < enh_valid_start)]
        enh_valid = train_data[train_data["date"] >= enh_valid_start]

        # calibデータが空の場合はvalidの前半を使う
        if enh_calib.empty:
            enh_calib = enh_valid.head(max(1, len(enh_valid) // 2))

        enhanced_models, enhanced_calibrator = train_enhanced(
            enh_train, enh_calib, enh_valid, enhanced_feature_cols, cat_features_enh
        )

    # テスト日の予測
    if test_date not in date_groups:
        continue
    day_df = date_groups[test_date]

    # Baseline 予測
    bl_proba = baseline_model.predict_proba(day_df[baseline_feature_cols])[:, 1]
    bl_result = evaluate_day(day_df, bl_proba, baseline_direction)
    if bl_result:
        bl_result["date"] = test_date
        baseline_results.append(bl_result)

    # Enhanced 予測 (アンサンブル平均 → キャリブレーション)
    enh_raw = np.column_stack(
        [m.predict_proba(day_df[enhanced_feature_cols])[:, 1] for m in enhanced_models]
    ).mean(axis=1)
    enh_proba = enhanced_calibrator.predict(enh_raw)
    enh_result = evaluate_day(day_df, enh_proba, enhanced_direction)
    if enh_result:
        enh_result["date"] = test_date
        enhanced_results.append(enh_result)

    if (i + 1) % 20 == 0:
        bl_done = len(baseline_results)
        enh_done = len(enhanced_results)
        bl_acc = sum(r["correct"] for r in baseline_results) / bl_done * 100 if bl_done else 0
        enh_acc = sum(r["correct"] for r in enhanced_results) / enh_done * 100 if enh_done else 0
        print(f"  進捗: {i+1}/{len(test_dates)}日  Baseline={bl_acc:.1f}%  Enhanced={enh_acc:.1f}%")

# === 結果表示 ===
bl_df = pd.DataFrame(baseline_results)
enh_df = pd.DataFrame(enhanced_results)

print()
print("=" * 85)
print("  Baseline vs Enhanced Walk-Forward Comparison Results")
print("=" * 85)
print()

bl_total = len(bl_df)
bl_correct = int(bl_df["correct"].sum())
bl_up_only = int(bl_df["up_correct"].sum())
bl_dn_only = int(bl_df["dn_correct"].sum())

enh_total = len(enh_df)
enh_correct = int(enh_df["correct"].sum())
enh_up_only = int(enh_df["up_correct"].sum())
enh_dn_only = int(enh_df["dn_correct"].sum())

print(f"【BASELINE】統合TOP1: {bl_correct}/{bl_total} = {bl_correct/bl_total*100:.1f}%  "
      f"上昇のみ: {bl_up_only/bl_total*100:.1f}%  値下がりのみ: {bl_dn_only/bl_total*100:.1f}%")
print(f"【ENHANCED】統合TOP1: {enh_correct}/{enh_total} = {enh_correct/enh_total*100:.1f}%  "
      f"上昇のみ: {enh_up_only/enh_total*100:.1f}%  値下がりのみ: {enh_dn_only/enh_total*100:.1f}%")

diff = enh_correct / enh_total * 100 - bl_correct / bl_total * 100
sign = "+" if diff >= 0 else ""
print(f"差分: {sign}{diff:.1f}pp")
print()

# Enhanced の方向別詳細
enh_up_days = enh_df[enh_df["direction"] == "UP"]
enh_dn_days = enh_df[enh_df["direction"] == "DOWN"]
print(f"【Enhanced 方向別】")
if len(enh_up_days) > 0:
    print(f"  UP採用:   {int(enh_up_days['correct'].sum())}/{len(enh_up_days)} = "
          f"{enh_up_days['correct'].mean()*100:.1f}%  ({len(enh_up_days)}日)")
if len(enh_dn_days) > 0:
    print(f"  DOWN採用: {int(enh_dn_days['correct'].sum())}/{len(enh_dn_days)} = "
          f"{enh_dn_days['correct'].mean()*100:.1f}%  ({len(enh_dn_days)}日)")
print()

# 月別比較
bl_df["month"] = pd.to_datetime(bl_df["date"]).dt.to_period("M").astype(str)
enh_df["month"] = pd.to_datetime(enh_df["date"]).dt.to_period("M").astype(str)

print(f"【月別比較】")
print(f"{'月':>10}  {'BL日数':>5}  {'BL正解率':>8}  {'ENH正解率':>9}  {'差分':>6}")
print("-" * 55)

all_months = sorted(set(bl_df["month"].unique()) | set(enh_df["month"].unique()))
for month in all_months:
    bl_m = bl_df[bl_df["month"] == month]
    enh_m = enh_df[enh_df["month"] == month]
    bl_days = len(bl_m)
    bl_rate = bl_m["correct"].mean() * 100 if bl_days > 0 else 0
    enh_days = len(enh_m)
    enh_rate = enh_m["correct"].mean() * 100 if enh_days > 0 else 0
    d = enh_rate - bl_rate
    s = "+" if d >= 0 else ""
    print(f"{month:>10}  {bl_days:>5}  {bl_rate:>7.1f}%  {enh_rate:>8.1f}%  {s}{d:>5.1f}pp")

print("-" * 55)
bl_rate_total = bl_correct / bl_total * 100
enh_rate_total = enh_correct / enh_total * 100
d_total = enh_rate_total - bl_rate_total
s_total = "+" if d_total >= 0 else ""
print(f"{'合計':>10}  {bl_total:>5}  {bl_rate_total:>7.1f}%  {enh_rate_total:>8.1f}%  {s_total}{d_total:>5.1f}pp")
print()
