"""全銘柄一括予測→ランキング出力（高値+5% / 安値-5%戦略）"""
import logging

import pandas as pd

import config
from src.model import load_model
from src.preprocessor import build_dataset

logger = logging.getLogger(__name__)


def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    for col in missing:
        df[col] = 0
    return df[feature_cols].fillna(0)


def predict_all(
    prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    dataset: pd.DataFrame | None = None,
) -> dict:
    """全銘柄の翌日予測を行い、2種類のランキングを返す。"""
    models, feature_cols = load_model()

    if dataset is not None:
        logger.info("学習済みデータセットを再利用")
        df = dataset
    else:
        logger.info("予測用特徴量を構築中...")
        df = build_dataset(prices, stock_list, index_data, pca_fit=False)

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    logger.info(f"予測対象日: {latest_date}, 銘柄数: {len(latest)}")

    if latest.empty:
        logger.warning("予測対象データがありません")
        return {"high": pd.DataFrame(), "low": pd.DataFrame(), "signals": []}

    X = _align_features(latest.copy(), feature_cols)
    latest["high_5pct_probability"] = models["target_high_5pct"].predict_proba(X)[:, 1]
    latest["low_5pct_probability"] = models["target_low_5pct"].predict_proba(X)[:, 1]

    vol_threshold = config.VOLUME_THRESHOLD
    if "volume" in latest.columns:
        filtered = latest[latest["volume"] >= vol_threshold].copy()
        logger.info(f"出来高フィルタ(>={vol_threshold:,}): {len(latest)} -> {len(filtered)}銘柄")
    else:
        filtered = latest.copy()

    name_map = stock_list.set_index("symbol")["name"].to_dict()
    sector_map = stock_list.set_index("symbol")["sector_name"].to_dict()
    filtered["name"] = filtered["symbol"].map(name_map)
    filtered["sector"] = filtered["symbol"].map(sector_map)

    keep_cols = ["symbol", "name", "sector", "high_5pct_probability", "low_5pct_probability", "volume"]
    keep_cols = [c for c in keep_cols if c in filtered.columns]

    high_ranking = filtered[keep_cols].sort_values("high_5pct_probability", ascending=False).reset_index(drop=True)
    high_ranking.index = high_ranking.index + 1
    high_ranking.index.name = "順位"

    low_ranking = filtered[keep_cols].sort_values("low_5pct_probability", ascending=False).reset_index(drop=True)
    low_ranking.index = low_ranking.index + 1
    low_ranking.index.name = "順位"

    signals = [
        {
            "type": "HIGH_5PCT",
            "confidence": float(high_ranking.iloc[0]["high_5pct_probability"]),
            "symbol": high_ranking.iloc[0]["symbol"],
            "name": high_ranking.iloc[0]["name"],
        },
        {
            "type": "LOW_5PCT",
            "confidence": float(low_ranking.iloc[0]["low_5pct_probability"]),
            "symbol": low_ranking.iloc[0]["symbol"],
            "name": low_ranking.iloc[0]["name"],
        },
    ]

    return {"high": high_ranking, "low": low_ranking, "signals": signals}


def predict_all_enhanced(
    prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    dataset: pd.DataFrame | None = None,
) -> dict:
    raise NotImplementedError("ENHANCED_MODE は新戦略に未対応です。`ENHANCED_MODE = False` を使用してください。")


def display_ranking(result: dict, top_n: int | None = None) -> None:
    """2種類のランキングをコンソールに表示する。"""
    top_n = top_n or config.RANKING_TOP_N
    signals = result.get("signals", [])
    high_ranking = result.get("high", pd.DataFrame())
    low_ranking = result.get("low", pd.DataFrame())

    if high_ranking.empty and low_ranking.empty:
        print("ランキングデータがありません。")
        return

    if signals:
        print(f"\n{'=' * 70}")
        print("  本日の注目シグナル")
        print(f"  高値+5%候補: {signals[0]['symbol']}  {signals[0]['name']}  ({signals[0]['confidence'] * 100:.1f}%)")
        print(f"  安値-5%候補: {signals[1]['symbol']}  {signals[1]['name']}  ({signals[1]['confidence'] * 100:.1f}%)")
        print(f"{'=' * 70}")

    print(f"\n{'=' * 70}")
    print(f"  翌日 安値が始値より5%以上低い確率ランキング (出来高>={config.VOLUME_THRESHOLD:,})")
    print(f"{'=' * 70}")
    print(f"{'順位':>4}  {'コード':<8} {'銘柄名':<20} {'業種':<14} {'-5%確率':>8}")
    print(f"{'-' * 70}")
    for i, row in low_ranking.head(top_n).iterrows():
        name = str(row.get("name", ""))[:18]
        sector = str(row.get("sector", ""))[:12]
        prob = row["low_5pct_probability"] * 100
        print(f"{i:>4}  {row['symbol']:<8} {name:<20} {sector:<14} {prob:>7.1f}%")
    print(f"{'-' * 70}")
    print(f"  全 {len(low_ranking)} 銘柄中 上位 {min(top_n, len(low_ranking))} 銘柄を表示")

    print(f"\n{'=' * 70}")
    print(f"  翌日 高値が始値より5%以上高い確率ランキング (出来高>={config.VOLUME_THRESHOLD:,})")
    print(f"{'=' * 70}")
    print(f"{'順位':>4}  {'コード':<8} {'銘柄名':<20} {'業種':<14} {'+5%確率':>8}")
    print(f"{'-' * 70}")
    for i, row in high_ranking.head(top_n).iterrows():
        name = str(row.get("name", ""))[:18]
        sector = str(row.get("sector", ""))[:12]
        prob = row["high_5pct_probability"] * 100
        print(f"{i:>4}  {row['symbol']:<8} {name:<20} {sector:<14} {prob:>7.1f}%")
    print(f"{'-' * 70}")
    print(f"  全 {len(high_ranking)} 銘柄中 上位 {min(top_n, len(high_ranking))} 銘柄を表示")
    print()
