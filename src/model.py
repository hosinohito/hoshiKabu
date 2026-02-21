"""LightGBM学習・評価・保存"""
import json
import logging
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import config
from src.preprocessor import get_feature_columns

logger = logging.getLogger(__name__)

META_PATH = config.MODELS_DIR / "train_meta.json"


def _save_meta(last_train_date: str) -> None:
    """学習メタ情報（最終学習日）を保存する。"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"last_train_date": last_train_date, "updated_at": datetime.now().isoformat()}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_meta() -> dict | None:
    """学習メタ情報を読み込む。"""
    if not META_PATH.exists():
        return None
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """時系列順にtrain/valid/testに分割する。"""
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df["date"].unique())

    n = len(dates)
    train_end = dates[int(n * config.TRAIN_RATIO)]
    valid_end = dates[int(n * (config.TRAIN_RATIO + config.VALID_RATIO))]

    train = df[df["date"] <= train_end]
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)]
    test = df[df["date"] > valid_end]

    logger.info(f"分割: train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test


def _evaluate(model, X, y, label: str) -> None:
    """モデルを評価してログ出力する。"""
    if len(y) == 0:
        return
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    ll = log_loss(y, proba)
    logger.info(f"[{label}] Accuracy={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
    print(f"[{label}] Accuracy={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")


def _align_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """不足列は0埋めし、余剰列は無視して整形する。"""
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = 0
    return df[feature_cols]


def _train_single_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_features: list[str],
    incremental: bool,
    existing_model: lgb.LGBMClassifier | None,
    last_date: str | None,
) -> lgb.LGBMClassifier:
    local_df = df.dropna(subset=[target_col]).copy()
    local_df[target_col] = local_df[target_col].astype(int)

    if incremental and existing_model is not None:
        new_data = local_df[local_df["date"] > last_date] if last_date else local_df
        if new_data.empty:
            logger.info(f"{target_col}: 新規データなし。追加学習をスキップ")
            return existing_model

        new_dates = np.sort(new_data["date"].unique())
        split_idx = int(len(new_dates) * 0.8)
        if split_idx == 0:
            split_idx = 1
        train_end = new_dates[min(split_idx, len(new_dates) - 1)]

        train_part = new_data[new_data["date"] <= train_end]
        valid_part = new_data[new_data["date"] > train_end]
        if valid_part.empty:
            valid_part = train_part.tail(max(1, len(train_part) // 5))

        X_train = _align_feature_columns(train_part.copy(), feature_cols).fillna(0)
        y_train = train_part[target_col]
        X_valid = _align_feature_columns(valid_part.copy(), feature_cols).fillna(0)
        y_valid = valid_part[target_col]

        inc_params = {
            **config.LIGHTGBM_PARAMS,
            "learning_rate": config.INCREMENTAL_LEARNING_RATE,
            "n_estimators": config.INCREMENTAL_N_ESTIMATORS,
        }
        inc_params.pop("early_stopping_rounds", None)
        model = lgb.LGBMClassifier(**inc_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            init_model=existing_model,
            callbacks=[lgb.log_evaluation(period=20), lgb.early_stopping(stopping_rounds=15)],
            categorical_feature=cat_features if cat_features else "auto",
        )
        _evaluate(model, X_valid, y_valid, f"{target_col}-incremental-valid")
        return model

    train, valid, test = time_series_split(local_df)
    X_train = _align_feature_columns(train.copy(), feature_cols).fillna(0)
    y_train = train[target_col]
    X_valid = _align_feature_columns(valid.copy(), feature_cols).fillna(0)
    y_valid = valid[target_col]
    X_test = _align_feature_columns(test.copy(), feature_cols).fillna(0)
    y_test = test[target_col]

    model = lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=config.LIGHTGBM_PARAMS.get("early_stopping_rounds", 30)),
        ],
        categorical_feature=cat_features if cat_features else "auto",
    )
    _evaluate(model, X_valid, y_valid, f"{target_col}-valid")
    _evaluate(model, X_test, y_test, f"{target_col}-test")
    return model


def train_model(df: pd.DataFrame, incremental: bool = False) -> dict[str, lgb.LGBMClassifier]:
    """2ターゲット（高値+5% / 安値-5%）のモデルを学習する。"""
    feature_cols = get_feature_columns(df)
    logger.info(f"特徴量数: {len(feature_cols)}")
    cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]
    targets = ["target_high_5pct", "target_low_5pct"]

    existing_models: dict[str, lgb.LGBMClassifier] = {}
    last_date = None
    if incremental:
        loaded_models, existing_cols = load_model()
        if set(existing_cols) != set(feature_cols):
            logger.warning("既存モデルと特徴量が不一致のため、既存モデル特徴量に合わせます。")
        feature_cols = existing_cols
        cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]
        existing_models = loaded_models
        meta = load_meta()
        last_date = meta["last_train_date"] if meta else None

    trained_models: dict[str, lgb.LGBMClassifier] = {}
    for target_col in targets:
        logger.info(f"{target_col} の学習を開始")
        trained_models[target_col] = _train_single_target(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            cat_features=cat_features,
            incremental=incremental,
            existing_model=existing_models.get(target_col),
            last_date=last_date,
        )

    max_date = str(pd.Timestamp(df["date"].max()).date())
    save_model(trained_models, feature_cols)
    _save_meta(max_date)
    return trained_models


def save_model(models: dict[str, lgb.LGBMClassifier], feature_cols: list[str]) -> None:
    """2モデルと特徴量リストを保存する。"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)
    logger.info(f"モデル保存: {model_path}")


def load_model() -> tuple[dict[str, lgb.LGBMClassifier], list[str]]:
    """保存済み2モデルを読み込む。"""
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("学習済みモデルが見つかりません。先にtrainを実行してください。")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    if "models" not in data:
        raise ValueError("旧形式モデルです。`python main.py train --full` を実行してください。")
    return data["models"], data["feature_cols"]


# === Enhanced (アンサンブル・キャリブレーション) ===


def compute_sample_weights(dates: np.ndarray, decay: float = None) -> np.ndarray:
    """指数減衰によるサンプル重みを計算する。最新日=1.0, 過去ほど小さい。"""
    decay = decay or config.SAMPLE_WEIGHT_DECAY
    unique_dates = np.sort(np.unique(dates))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    max_idx = len(unique_dates) - 1
    indices = np.array([date_to_idx[d] for d in dates])
    weights = decay ** (max_idx - indices)
    return weights


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
    sample_weights: np.ndarray | None = None,
    ensemble_params: list[dict] | None = None,
    common_params: dict | None = None,
) -> list[lgb.LGBMClassifier]:
    """アンサンブル用の複数LightGBMモデルを学習する。"""
    common = common_params or config.ENSEMBLE_COMMON_PARAMS
    model_configs = ensemble_params or config.ENSEMBLE_MODELS
    models = []

    for i, params in enumerate(model_configs):
        merged = {**common, **params}
        model = lgb.LGBMClassifier(**merged)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.log_evaluation(period=999), lgb.early_stopping(stopping_rounds=20)],
            categorical_feature=cat_features or "auto",
            sample_weight=sample_weights,
        )
        models.append(model)
        logger.info(f"アンサンブルモデル {i+1}/{len(model_configs)} 学習完了")

    return models


def ensemble_predict_proba(models: list[lgb.LGBMClassifier], X: pd.DataFrame) -> np.ndarray:
    """アンサンブルモデルの予測確率平均を返す。"""
    probas = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return probas.mean(axis=1)


def fit_calibrator(
    raw_proba: np.ndarray, y_true: np.ndarray, method: str = None
) -> IsotonicRegression:
    """Isotonic regressionでキャリブレーターを学習する。"""
    method = method or config.CALIBRATION_METHOD
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(raw_proba, y_true)
    logger.info("キャリブレーター学習完了")
    return calibrator


def save_model_enhanced(
    models: list[lgb.LGBMClassifier],
    calibrator: IsotonicRegression,
    feature_cols: list[str],
) -> None:
    """アンサンブルモデル+キャリブレーターを保存する。"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / "lgbm_enhanced.pkl"
    with open(path, "wb") as f:
        pickle.dump({"models": models, "calibrator": calibrator, "feature_cols": feature_cols}, f)
    logger.info(f"Enhanced モデル保存: {path}")


def load_model_enhanced() -> tuple[list[lgb.LGBMClassifier], IsotonicRegression, list[str]]:
    """アンサンブルモデル+キャリブレーターを読み込む。"""
    path = config.MODELS_DIR / "lgbm_enhanced.pkl"
    if not path.exists():
        raise FileNotFoundError("Enhanced モデルが見つかりません。")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["models"], data["calibrator"], data["feature_cols"]
