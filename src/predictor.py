"""全銘柄一括予測→ランキング出力（上昇・値下がり統合戦略）"""
import logging

import pandas as pd

import config
from src.model import load_model
from src.preprocessor import build_dataset, get_feature_columns

logger = logging.getLogger(__name__)


def predict_all(
    prices: pd.DataFrame,
    stock_list: pd.DataFrame,
    index_data: pd.DataFrame,
    dataset: pd.DataFrame | None = None,
) -> dict:
    """全銘柄の翌日予測を行い、上昇・値下がりランキングと推奨シグナルを返す。"""
    model, feature_cols = load_model()

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
        return {"up": pd.DataFrame(), "down": pd.DataFrame(), "signal": None}

    # 特徴量の整合性
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
    X = latest[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    latest["up_probability"] = proba
    latest["down_probability"] = 1 - proba

    # 出来高フィルタ
    vol_threshold = config.VOLUME_THRESHOLD
    if "volume" in latest.columns:
        filtered = latest[latest["volume"] >= vol_threshold].copy()
        logger.info(f"出来高フィルタ(>={vol_threshold:,}): {len(latest)} -> {len(filtered)}銘柄")
    else:
        filtered = latest.copy()

    # 銘柄情報を結合
    name_map = stock_list.set_index("symbol")["name"].to_dict()
    sector_map = stock_list.set_index("symbol")["sector_name"].to_dict()
    for target_df in [filtered]:
        target_df["name"] = target_df["symbol"].map(name_map)
        target_df["sector"] = target_df["symbol"].map(sector_map)

    keep_cols = ["symbol", "name", "sector", "up_probability", "down_probability", "volume"]
    keep_cols = [c for c in keep_cols if c in filtered.columns]

    # 上昇ランキング
    up_ranking = (
        filtered[keep_cols]
        .sort_values("up_probability", ascending=False)
        .reset_index(drop=True)
    )
    up_ranking.index = up_ranking.index + 1
    up_ranking.index.name = "順位"

    # 値下がりランキング
    down_ranking = (
        filtered[keep_cols]
        .sort_values("down_probability", ascending=False)
        .reset_index(drop=True)
    )
    down_ranking.index = down_ranking.index + 1
    down_ranking.index.name = "順位"

    # 統合シグナル: 上昇TOP1 vs 値下がりTOP1の確信度比較
    up_top1_conf = filtered["up_probability"].max()
    dn_top1_conf = filtered["down_probability"].max()

    if up_top1_conf >= dn_top1_conf:
        signal = {
            "direction": "UP",
            "confidence": up_top1_conf,
            "symbol": up_ranking.iloc[0]["symbol"],
            "name": up_ranking.iloc[0]["name"],
        }
    else:
        signal = {
            "direction": "DOWN",
            "confidence": dn_top1_conf,
            "symbol": down_ranking.iloc[0]["symbol"],
            "name": down_ranking.iloc[0]["name"],
        }

    return {"up": up_ranking, "down": down_ranking, "signal": signal}


def display_ranking(result: dict, top_n: int | None = None) -> None:
    """統合ランキングをコンソールに表示する。"""
    top_n = top_n or config.RANKING_TOP_N
    signal = result.get("signal")
    up_ranking = result.get("up", pd.DataFrame())
    down_ranking = result.get("down", pd.DataFrame())

    if up_ranking.empty and down_ranking.empty:
        print("ランキングデータがありません。")
        return

    # 統合シグナル
    if signal:
        direction_jp = "上昇" if signal["direction"] == "UP" else "値下がり"
        print(f"\n{'=' * 70}")
        print(f"  本日の推奨シグナル: {direction_jp}")
        print(f"  銘柄: {signal['symbol']}  {signal['name']}")
        print(f"  確信度: {signal['confidence'] * 100:.1f}%")
        print(f"{'=' * 70}")

    # 値下がりランキング
    print(f"\n{'=' * 70}")
    print(f"  翌日値下がり確率ランキング (出来高>={config.VOLUME_THRESHOLD:,})")
    print(f"{'=' * 70}")
    print(f"{'順位':>4}  {'コード':<8} {'銘柄名':<20} {'業種':<14} {'値下がり確率':>8}")
    print(f"{'-' * 70}")
    for i, row in down_ranking.head(top_n).iterrows():
        name = str(row.get("name", ""))[:18]
        sector = str(row.get("sector", ""))[:12]
        prob = row["down_probability"] * 100
        print(f"{i:>4}  {row['symbol']:<8} {name:<20} {sector:<14} {prob:>7.1f}%")
    print(f"{'-' * 70}")
    print(f"  全 {len(down_ranking)} 銘柄中 上位 {min(top_n, len(down_ranking))} 銘柄を表示")

    # 上昇ランキング
    print(f"\n{'=' * 70}")
    print(f"  翌日上昇確率ランキング (出来高>={config.VOLUME_THRESHOLD:,})")
    print(f"{'=' * 70}")
    print(f"{'順位':>4}  {'コード':<8} {'銘柄名':<20} {'業種':<14} {'上昇確率':>8}")
    print(f"{'-' * 70}")
    for i, row in up_ranking.head(top_n).iterrows():
        name = str(row.get("name", ""))[:18]
        sector = str(row.get("sector", ""))[:12]
        prob = row["up_probability"] * 100
        print(f"{i:>4}  {row['symbol']:<8} {name:<20} {sector:<14} {prob:>7.1f}%")
    print(f"{'-' * 70}")
    print(f"  全 {len(up_ranking)} 銘柄中 上位 {min(top_n, len(up_ranking))} 銘柄を表示")
    print()
