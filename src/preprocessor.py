"""PCA市場ファクター抽出・個別銘柄特徴量・データセット構築"""
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import config

logger = logging.getLogger(__name__)


def compute_intraday_return(prices: pd.DataFrame) -> pd.DataFrame:
    """日中リターン (close - open) / open を計算する。"""
    df = prices.copy()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    return df


def compute_individual_features(prices: pd.DataFrame) -> pd.DataFrame:
    """個別銘柄の特徴量を生成する（vectorized版）。"""
    df = prices.sort_values(["symbol", "date"]).copy()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    g = df.groupby("symbol")

    # 前日シフトを事前に1回だけ計算
    intraday_shifted = g["intraday_return"].shift(1)
    daily_shifted = g["daily_return"].shift(1)
    close_shifted = g["close"].shift(1)

    for n in config.LOOKBACK_DAYS:
        # 過去N日の日中リターン平均（vectorized rolling）
        df[f"intraday_ret_ma{n}"] = intraday_shifted.groupby(
            df["symbol"]
        ).rolling(n, min_periods=1).mean().droplevel(0)
        # 過去N日のボラティリティ（vectorized rolling）
        df[f"volatility_{n}d"] = daily_shifted.groupby(
            df["symbol"]
        ).rolling(n, min_periods=1).std().droplevel(0)
        # 過去N日の終値移動平均乖離率（vectorized rolling）
        close_ma = close_shifted.groupby(
            df["symbol"]
        ).rolling(n, min_periods=1).mean().droplevel(0)
        df[f"ma{n}_deviation"] = df["close"] / close_ma - 1

    # 出来高変化率
    if "volume" in df.columns:
        df["volume_change"] = g["volume"].pct_change().shift(1)
        vol_shifted = g["volume"].shift(1)
        vol_ma5 = vol_shifted.groupby(
            df["symbol"]
        ).rolling(5, min_periods=1).mean().droplevel(0)
        df["volume_ma5_ratio"] = df["volume"] / vol_ma5

    return df


def compute_market_features(index_data: pd.DataFrame) -> pd.DataFrame:
    """市場指数の前日リターンを計算する。"""
    if index_data.empty:
        logger.warning("指数データが空です")
        return pd.DataFrame()

    records = []
    for sym in index_data["symbol"].unique():
        sub = index_data[index_data["symbol"] == sym].sort_values("date").copy()
        prefix = "nikkei" if "N225" in sym or "N225" in str(sym) else "topix"
        sub[f"{prefix}_return"] = sub["close"].pct_change()
        sub[f"{prefix}_intraday"] = (sub["close"] - sub["open"]) / sub["open"]
        records.append(sub[["date", f"{prefix}_return", f"{prefix}_intraday"]].dropna())

    if not records:
        return pd.DataFrame()

    result = records[0]
    for r in records[1:]:
        result = result.merge(r, on="date", how="outer")

    result = result.sort_values("date").reset_index(drop=True)
    return result


def compute_pca_factors(
    prices: pd.DataFrame, n_components: int | None = None, fit: bool = True
) -> tuple[pd.DataFrame, PCA | None]:
    """全銘柄の前日リターンからPCAで市場ファクターを抽出する。"""
    n_components = n_components or config.PCA_N_COMPONENTS
    pca_model_path = config.DATA_PROCESSED_DIR / config.PCA_MODEL_CACHE

    df = prices[["date", "symbol", "close"]].copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    pivot = df.pivot_table(index="date", columns="symbol", values="daily_return")
    # 欠損が多い銘柄を除外（80%以上データがある銘柄のみ）
    threshold = len(pivot) * 0.8
    pivot = pivot.dropna(axis=1, thresh=int(threshold))
    pivot = pivot.fillna(0)

    # PCA成分数をデータに合わせて調整
    actual_components = min(n_components, pivot.shape[1], pivot.shape[0])
    if actual_components < 1:
        logger.warning("PCA用のデータが不足しています")
        return pd.DataFrame(), None

    if fit:
        pca = PCA(n_components=actual_components)
        factors = pca.fit_transform(pivot.values)
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(pca_model_path, "wb") as f:
            pickle.dump(pca, f)
        logger.info(f"PCA学習完了: {actual_components}成分, 寄与率合計={pca.explained_variance_ratio_.sum():.3f}")
    else:
        if not pca_model_path.exists():
            raise FileNotFoundError("学習済みPCAモデルが見つかりません。先にtrainを実行してください。")
        with open(pca_model_path, "rb") as f:
            pca = pickle.load(f)
        # 列数をPCA学習時と合わせる
        n_features = pca.n_features_in_
        if pivot.shape[1] > n_features:
            pivot = pivot.iloc[:, :n_features]
        elif pivot.shape[1] < n_features:
            for i in range(pivot.shape[1], n_features):
                pivot[f"__pad_{i}"] = 0
        factors = pca.transform(pivot.values)

    factor_cols = [f"pca_factor_{i}" for i in range(factors.shape[1])]
    factor_df = pd.DataFrame(factors, index=pivot.index, columns=factor_cols)
    factor_df = factor_df.reset_index().rename(columns={"index": "date"})
    # 1日シフト（前日のファクターを使う）
    for col in factor_cols:
        factor_df[col] = factor_df[col].shift(1)
    factor_df = factor_df.dropna()

    return factor_df, pca


def build_dataset(
    prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    pca_fit: bool = True,
) -> pd.DataFrame:
    """全特徴量を結合してデータセットを構築する。"""
    logger.info("個別銘柄特徴量を生成中...")
    df = compute_individual_features(prices)

    # ターゲット: 翌日の始値→終値が上昇なら1
    df["next_open"] = df.groupby("symbol")["open"].shift(-1)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["target"] = (df["next_close"] > df["next_open"]).astype(int)

    # 市場指数特徴量
    logger.info("市場指数特徴量を結合中...")
    market_feat = compute_market_features(index_data)
    if not market_feat.empty:
        # 市場特徴量は1日シフト（前日の値を使用）
        for col in market_feat.columns:
            if col != "date":
                market_feat[col] = market_feat[col].shift(1)
        market_feat = market_feat.dropna()
        df = df.merge(market_feat, on="date", how="left")

    # PCAファクター
    logger.info("PCA市場ファクターを抽出中...")
    pca_factors, _ = compute_pca_factors(prices, fit=pca_fit)
    if not pca_factors.empty:
        df = df.merge(pca_factors, on="date", how="left")

    # セクター情報
    if stock_list is not None and "sector_code" in stock_list.columns:
        sector_map = stock_list.set_index("symbol")[["sector_code", "sector_name"]].to_dict("index")
        df["sector_code"] = df["symbol"].map(lambda s: sector_map.get(s, {}).get("sector_code", "0"))
        le = LabelEncoder()
        df["sector_encoded"] = le.fit_transform(df["sector_code"].astype(str))
        # LabelEncoderを保存
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.DATA_PROCESSED_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    # 曜日・月
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    logger.info(f"データセット構築完了: {len(df)}行, {df['symbol'].nunique()}銘柄")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """特徴量カラムのリストを返す。"""
    exclude = {
        "date", "symbol", "code", "name", "sector_code", "sector_name",
        "market", "open", "high", "low", "close", "volume",
        "next_open", "next_close", "target",
    }
    return [c for c in df.columns if c not in exclude]
