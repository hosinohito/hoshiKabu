"""PCA市場ファクター抽出・個別銘柄特徴量・データセット構築"""
import logging
import pickle
from pathlib import Path

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

    # groupbyオブジェクトを事前に作成して使い回す
    intraday_grp = intraday_shifted.groupby(df["symbol"])
    daily_grp = daily_shifted.groupby(df["symbol"])
    close_grp = close_shifted.groupby(df["symbol"])

    for n in config.LOOKBACK_DAYS:
        # 過去N日の日中リターン平均（vectorized rolling）
        df[f"intraday_ret_ma{n}"] = intraday_grp.rolling(n, min_periods=1).mean().droplevel(0)
        # 過去N日のボラティリティ（vectorized rolling）
        df[f"volatility_{n}d"] = daily_grp.rolling(n, min_periods=1).std().droplevel(0)
        # 過去N日の終値移動平均乖離率（vectorized rolling）
        close_ma = close_grp.rolling(n, min_periods=1).mean().droplevel(0)
        df[f"ma{n}_deviation"] = df["close"] / close_ma - 1

    # 出来高変化率
    if "volume" in df.columns:
        df["volume_change"] = g["volume"].pct_change().shift(1)
        vol_shifted = g["volume"].shift(1)
        vol_grp = vol_shifted.groupby(df["symbol"])
        vol_ma5 = vol_grp.rolling(5, min_periods=1).mean().droplevel(0)
        df["volume_ma5_ratio"] = df["volume"] / vol_ma5

    # --- 新テクニカル指標 (shift(1)済みcloseで計算しリーク回避) ---

    # RSI (14日)
    close_diff = close_shifted.groupby(df["symbol"]).diff()
    gain = close_diff.clip(lower=0)
    loss = (-close_diff).clip(lower=0)
    avg_gain = gain.groupby(df["symbol"]).rolling(config.RSI_PERIOD, min_periods=1).mean().droplevel(0)
    avg_loss = loss.groupby(df["symbol"]).rolling(config.RSI_PERIOD, min_periods=1).mean().droplevel(0)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD (shift(1)済みcloseベース)
    ema_fast = close_shifted.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=config.MACD_FAST, min_periods=1).mean()
    )
    ema_slow = close_shifted.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=config.MACD_SLOW, min_periods=1).mean()
    )
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=config.MACD_SIGNAL, min_periods=1).mean()
    )
    df["macd_histogram"] = macd_line - signal_line
    df["macd_normalized"] = df["macd_histogram"] / close_shifted.replace(0, np.nan)

    # Bollinger Bands (shift(1)済みcloseベース)
    bb_ma = close_grp.rolling(config.BOLLINGER_PERIOD, min_periods=1).mean().droplevel(0)
    bb_std = close_grp.rolling(config.BOLLINGER_PERIOD, min_periods=1).std().droplevel(0)
    bb_upper = bb_ma + config.BOLLINGER_STD * bb_std
    bb_lower = bb_ma - config.BOLLINGER_STD * bb_std
    bb_width = bb_upper - bb_lower
    df["bollinger_position"] = (close_shifted - bb_lower) / bb_width.replace(0, np.nan)
    df["bollinger_width"] = bb_width / bb_ma.replace(0, np.nan)

    # 出来高スパイク (shift(1)済みvolumeベース)
    if "volume" in df.columns:
        vol_spike_ma = vol_grp.rolling(config.VOLUME_SPIKE_WINDOW, min_periods=1).mean().droplevel(0)
        df["volume_spike_ratio"] = vol_shifted / vol_spike_ma.replace(0, np.nan)

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

    pivot_cache_path = config.DATA_PROCESSED_DIR / config.PCA_PIVOT_CACHE

    df = prices[["date", "symbol", "close"]].copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    if fit:
        # fit時: pivotを計算してキャッシュ保存
        pivot = df.pivot_table(index="date", columns="symbol", values="daily_return")
        threshold = len(pivot) * 0.8
        pivot = pivot.dropna(axis=1, thresh=int(threshold))
        pivot = pivot.fillna(0)
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pivot.to_parquet(pivot_cache_path)
        logger.info(f"PCA pivot キャッシュ保存: {pivot.shape}")
    else:
        # predict時: キャッシュがあれば新規日付分だけ追記
        if pivot_cache_path.exists():
            cached_pivot = pd.read_parquet(pivot_cache_path)
            cached_dates = set(cached_pivot.index)
            new_dates = df[~df["date"].isin(cached_dates)]
            if not new_dates.empty:
                new_pivot = new_dates.pivot_table(index="date", columns="symbol", values="daily_return")
                new_pivot = new_pivot.reindex(columns=cached_pivot.columns, fill_value=0).fillna(0)
                pivot = pd.concat([cached_pivot, new_pivot]).sort_index()
            else:
                pivot = cached_pivot
            logger.info(f"PCA pivot キャッシュ利用 (cached={len(cached_dates)}, total={len(pivot)})")
        else:
            pivot = df.pivot_table(index="date", columns="symbol", values="daily_return")
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


def _label_encoder_path() -> Path:
    return config.DATA_PROCESSED_DIR / "label_encoder.pkl"


def _load_label_encoder() -> LabelEncoder | None:
    path = _label_encoder_path()
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_label_encoder(le: LabelEncoder) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(_label_encoder_path(), "wb") as f:
        pickle.dump(le, f)


def build_dataset(
    prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    pca_fit: bool = True,
    fit_encoders: bool | None = None,
) -> pd.DataFrame:
    """全特徴量を結合してデータセットを構築する。"""
    # ルックバック期間で絞り込み（メモリ削減）
    if config.TRAIN_LOOKBACK_YEARS is not None and not prices.empty:
        max_date = pd.to_datetime(prices["date"]).max()
        cutoff = max_date - pd.DateOffset(years=config.TRAIN_LOOKBACK_YEARS)
        prices = prices[pd.to_datetime(prices["date"]) >= cutoff]
        logger.info(f"学習データを直近{config.TRAIN_LOOKBACK_YEARS}年に絞り込み: {len(prices)}行")

    logger.info("個別銘柄特徴量を生成中...")
    df = compute_individual_features(prices)

    # ターゲット:
    # 1) 翌日の高値が始値より5%以上高い
    # 2) 翌日の安値が始値より5%以上低い
    df["next_open"] = df.groupby("symbol")["open"].shift(-1)
    df["next_high"] = df.groupby("symbol")["high"].shift(-1)
    df["next_low"] = df.groupby("symbol")["low"].shift(-1)
    df["target_high_5pct"] = (df["next_high"] >= df["next_open"] * 1.05).astype(int)
    df["target_low_5pct"] = (df["next_low"] <= df["next_open"] * 0.95).astype(int)

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
        sector_code_map = {s: v.get("sector_code", "0") for s, v in sector_map.items()}
        df["sector_code"] = df["symbol"].map(sector_code_map).fillna("0")
        fit_encoders = pca_fit if fit_encoders is None else fit_encoders
        le = None
        if not fit_encoders:
            le = _load_label_encoder()
        if le is None:
            le = LabelEncoder()
            df["sector_encoded"] = le.fit_transform(df["sector_code"].astype(str))
            _save_label_encoder(le)
        else:
            df["sector_encoded"] = le.transform(df["sector_code"].astype(str))

    # セクター相対強度 (shift(1)済みdaily_returnベース)
    if "sector_code" in df.columns and "daily_return" in df.columns:
        shifted_ret = df.groupby("symbol")["daily_return"].shift(1)
        sector_mean = shifted_ret.groupby(df["sector_code"]).transform("mean")
        df["sector_relative_strength"] = shifted_ret - sector_mean

    # 曜日・月
    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    # float64 → float32 でメモリ使用量を約50%削減
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    logger.info(f"データセット構築完了: {len(df)}行, {df['symbol'].nunique()}銘柄, "
                f"メモリ={df.memory_usage(deep=True).sum() / 1024**3:.2f}GB")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """特徴量カラムのリストを返す。"""
    exclude = {
        "date", "symbol", "code", "name", "sector_code", "sector_name",
        "market", "open", "high", "low", "close", "volume",
        "next_open", "next_high", "next_low",
        "target", "target_high_5pct", "target_low_5pct",
    }
    return [c for c in df.columns if c not in exclude]
