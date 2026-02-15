"""設定値"""
from pathlib import Path

# Enhanced モード (アンサンブル・キャリブレーション・非対称閾値)
# True にすると train/predict で Enhanced パイプラインを使用
ENHANCED_MODE = False

# ディレクトリ
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# データ取得
YFINANCE_BATCH_SIZE = 500
YFINANCE_SLEEP_SEC = 1.0
YFINANCE_MAX_RETRIES = 3
DATA_START_DATE = "2020-01-01"

# JPX銘柄リスト
JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# 市場指数シンボル
NIKKEI225_SYMBOL = "^N225"
TOPIX_SYMBOL = "1306.T"  # TOPIX連動ETFで代替

# 特徴量パラメータ
LOOKBACK_DAYS = [1, 2, 3, 4, 5, 6]
PCA_N_COMPONENTS = 50

# テクニカル指標
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
VOLUME_SPIKE_WINDOW = 20

# 非対称閾値
UP_THRESHOLD = 0.52
DOWN_THRESHOLD = 0.55

# サンプル重み付け (0.998^250日≈0.61)
SAMPLE_WEIGHT_DECAY = 0.998

# キャリブレーション
CALIBRATION_METHOD = "isotonic"

# モデルパラメータ
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "is_unbalance": True,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}

# 時系列分割
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
# TEST_RATIO = 0.15 (残り)

# ランキング表示
RANKING_TOP_N = 50
VOLUME_THRESHOLD = 10000  # 出来高フィルタ閾値

# 追加学習
INCREMENTAL_N_ESTIMATORS = 100  # 追加学習時のブースティング回数
INCREMENTAL_LEARNING_RATE = 0.02  # 追加学習時は小さめの学習率

# アンサンブル共通パラメータ
ENSEMBLE_COMMON_PARAMS = {
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

# アンサンブル個別パラメータ (3モデル)
ENSEMBLE_MODELS = [
    {  # A: 標準
        "num_leaves": 63, "learning_rate": 0.05, "n_estimators": 500,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "min_child_samples": 50,
    },
    {  # B: 浅く速く
        "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "min_child_samples": 100,
    },
    {  # C: 深く正則化
        "num_leaves": 127, "learning_rate": 0.03, "n_estimators": 600,
        "feature_fraction": 0.6, "bagging_fraction": 0.9, "min_child_samples": 30,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
]

# キャッシュファイル名
STOCK_LIST_CACHE = "stock_list.parquet"
PRICE_CACHE_PREFIX = "prices_"
FEATURES_CACHE = "features.parquet"
PCA_MODEL_CACHE = "pca_model.pkl"
PCA_PIVOT_CACHE = "pca_pivot_cache.parquet"
