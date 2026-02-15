"""LightGBM学習・評価・保存"""
import json
import logging
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
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


def train_model(df: pd.DataFrame, incremental: bool = False) -> lgb.LGBMClassifier:
    """LightGBMモデルを学習する。

    Args:
        df: 特徴量付きデータセット
        incremental: Trueなら既存モデルをベースに追加学習
    """
    feature_cols = get_feature_columns(df)
    logger.info(f"特徴量数: {len(feature_cols)}")

    # ターゲットが無い行を除外
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]

    # --- 追加学習モード ---
    if incremental:
        existing_model, existing_cols = load_model()
        meta = load_meta()
        last_date = meta["last_train_date"] if meta else None

        if last_date:
            # 前回学習日以降のデータで追加学習
            new_data = df[df["date"] > last_date]
            if new_data.empty:
                logger.info("新規データなし。追加学習をスキップ")
                print("新規データなし。追加学習をスキップ")
                return existing_model
            logger.info(f"追加学習: {last_date} 以降の {len(new_data)} サンプルで継続学習")
            print(f"追加学習: {last_date} 以降の {len(new_data)} サンプル")
        else:
            new_data = df

        # 新データの80%を学習、20%を検証に使う
        new_dates = np.sort(new_data["date"].unique())
        split_idx = int(len(new_dates) * 0.8)
        if split_idx == 0:
            split_idx = 1
        train_end = new_dates[min(split_idx, len(new_dates) - 1)]

        train_part = new_data[new_data["date"] <= train_end]
        valid_part = new_data[new_data["date"] > train_end]
        if valid_part.empty:
            valid_part = train_part.tail(max(1, len(train_part) // 5))

        X_train = train_part[feature_cols]
        y_train = train_part["target"]
        X_valid = valid_part[feature_cols]
        y_valid = valid_part["target"]

        # 既存モデルのBoosterをinit_modelとして渡して継続学習
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
            callbacks=[
                lgb.log_evaluation(period=20),
                lgb.early_stopping(stopping_rounds=15),
            ],
            categorical_feature=cat_features if cat_features else "auto",
        )

        _evaluate(model, X_valid, y_valid, "incremental-valid")

        max_date = str(pd.Timestamp(df["date"].max()).date())
        save_model(model, feature_cols)
        _save_meta(max_date)
        return model

    # --- フル学習モード ---
    train, valid, test = time_series_split(df)

    X_train = train[feature_cols]
    y_train = train["target"]
    X_valid = valid[feature_cols]
    y_valid = valid["target"]
    X_test = test[feature_cols]
    y_test = test["target"]

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

    _evaluate(model, X_valid, y_valid, "valid")
    _evaluate(model, X_test, y_test, "test")

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== 特徴量重要度 TOP20 ===")
    print(importance.head(20).to_string(index=False))

    max_date = str(pd.Timestamp(df["date"].max()).date())
    save_model(model, feature_cols)
    _save_meta(max_date)
    return model


def save_model(model: lgb.LGBMClassifier, feature_cols: list[str]) -> None:
    """モデルと特徴量リストを保存する。"""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    logger.info(f"モデル保存: {model_path}")


def load_model() -> tuple[lgb.LGBMClassifier, list[str]]:
    """保存済みモデルを読み込む。"""
    model_path = config.MODELS_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("学習済みモデルが見つかりません。先にtrainを実行してください。")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]
