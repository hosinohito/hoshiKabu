"""CLI: fetch / train / predict / run"""
import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from src.fetcher import fetch_stock_list, fetch_price_data, fetch_index_data
from src.preprocessor import build_dataset
from src.model import train_model, load_meta
from src.predictor import predict_all, display_ranking

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool = False) -> None:
    """ログ設定。通常時はyfinance等の外部ライブラリのノイズを抑制する。"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not debug:
        # yfinanceの"possibly delisted"等のエラーログを抑制
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        # LightGBMの冗長なログを抑制
        logging.getLogger("lightgbm").setLevel(logging.WARNING)


def _ensure_dirs() -> None:
    for d in [config.DATA_RAW_DIR, config.DATA_PROCESSED_DIR, config.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_path = config.DATA_RAW_DIR / "prices.parquet"
    index_path = config.DATA_RAW_DIR / "index_prices.parquet"
    if not prices_path.exists():
        print("価格データが見つかりません。先に fetch を実行してください。")
        sys.exit(1)
    prices = pd.read_parquet(prices_path)
    index_data = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    return prices, index_data


def cmd_fetch(args: argparse.Namespace) -> None:
    """全銘柄データを取得する。"""
    logger.info("=== データ取得開始 ===")
    _ensure_dirs()

    stock_list = fetch_stock_list(use_cache=not args.refresh)
    print(f"銘柄リスト: {len(stock_list)}銘柄")

    index_data = fetch_index_data()
    print(f"指数データ: {len(index_data)}行")

    prices = fetch_price_data(stock_list, use_cache=not args.refresh)
    print(f"価格データ: {len(prices)}行, {prices['symbol'].nunique() if not prices.empty else 0}銘柄")

    logger.info("=== データ取得完了 ===")


def _train_enhanced(dataset: pd.DataFrame) -> None:
    """Enhanced モードの学習: アンサンブル3モデル + キャリブレーション。"""
    from src.model import (
        compute_sample_weights, train_ensemble, fit_calibrator,
        ensemble_predict_proba, save_model_enhanced,
    )
    from src.preprocessor import get_feature_columns
    import numpy as np

    feature_cols = get_feature_columns(dataset)
    df = dataset.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    cat_features = [c for c in feature_cols if c in ("sector_encoded", "dayofweek", "month")]

    dates = np.sort(df["date"].unique())
    n = len(dates)
    calib_start = dates[int(n * config.TRAIN_RATIO)]
    valid_start = dates[int(n * (config.TRAIN_RATIO + 0.05))]
    test_start = dates[int(n * (config.TRAIN_RATIO + config.VALID_RATIO))]

    train_data = df[df["date"] <= calib_start]
    calib_data = df[(df["date"] > calib_start) & (df["date"] <= valid_start)]
    valid_data = df[(df["date"] > valid_start) & (df["date"] <= test_start)]

    if calib_data.empty:
        calib_data = valid_data.head(max(1, len(valid_data) // 2))

    weights = compute_sample_weights(train_data["date"].values)

    logger.info("アンサンブルモデル学習中...")
    models = train_ensemble(
        train_data[feature_cols], train_data["target"],
        valid_data[feature_cols], valid_data["target"],
        cat_features, sample_weights=weights,
    )

    logger.info("キャリブレーション学習中...")
    calib_proba = ensemble_predict_proba(models, calib_data[feature_cols])
    calibrator = fit_calibrator(calib_proba, calib_data["target"].values)

    save_model_enhanced(models, calibrator, feature_cols)
    logger.info("Enhanced モデル学習完了")


def cmd_train(args: argparse.Namespace) -> None:
    """モデルを学習する。"""
    incremental = not args.full
    mode = "フル学習" if args.full else "追加学習"
    logger.info(f"=== モデル{mode}開始 ===")

    # 追加学習なのに既存モデルが無い場合はフル学習にフォールバック
    if incremental and load_meta() is None:
        logger.info("既存モデルなし。フル学習に切り替え")
        incremental = False

    stock_list = fetch_stock_list()
    prices, index_data = _load_prices()

    logger.info("データセット構築中...")
    dataset = build_dataset(prices, stock_list, index_data, pca_fit=True)

    if config.ENHANCED_MODE:
        raise NotImplementedError("ENHANCED_MODE は新戦略に未対応です。`ENHANCED_MODE = False` を使用してください。")
    logger.info("モデル学習中...")
    train_model(dataset, incremental=incremental)

    logger.info(f"=== モデル{mode}完了 ===")


def cmd_predict(args: argparse.Namespace) -> None:
    """全銘柄の翌日 高値+5%% / 安値-5%% 確率を予測する。"""
    logger.info("=== 予測開始 ===")

    stock_list = fetch_stock_list()
    prices, index_data = _load_prices()

    if config.ENHANCED_MODE:
        raise NotImplementedError("ENHANCED_MODE は新戦略に未対応です。`ENHANCED_MODE = False` を使用してください。")
    result = predict_all(prices, stock_list, index_data)
    display_ranking(result, top_n=args.top)

    if args.output:
        result["low"].to_csv(args.output, encoding="utf-8-sig")
        print(f"ランキングをCSV出力: {args.output}")

    logger.info("=== 予測完了 ===")


def cmd_run(args: argparse.Namespace) -> None:
    """fetch → 追加学習 → predict をワンショットで実行する。"""
    logger.info("=" * 60)
    logger.info("  ワンショット実行: fetch → train → predict")
    logger.info("=" * 60)
    _ensure_dirs()

    # 1. データ取得（差分）
    logger.info("--- [1/3] データ取得 ---")
    stock_list = fetch_stock_list()
    print(f"銘柄リスト: {len(stock_list)}銘柄")

    index_data = fetch_index_data()
    prices = fetch_price_data(stock_list)
    print(f"価格データ: {len(prices)}行, {prices['symbol'].nunique() if not prices.empty else 0}銘柄")

    # 2. 学習（既存モデルがあれば追加学習、なければフル学習）
    logger.info("--- [2/3] モデル学習 ---")
    incremental = load_meta() is not None
    mode = "追加学習" if incremental else "フル学習（初回）"
    print(f"学習モード: {mode}")

    dataset = build_dataset(prices, stock_list, index_data, pca_fit=True)
    if config.ENHANCED_MODE:
        raise NotImplementedError("ENHANCED_MODE は新戦略に未対応です。`ENHANCED_MODE = False` を使用してください。")
    train_model(dataset, incremental=incremental)

    # 3. 予測（学習時のdatasetを再利用してbuild_datasetの二重呼び出しを回避）
    logger.info("--- [3/3] 予測・ランキング ---")
    result = predict_all(prices, stock_list, index_data, dataset=dataset)
    display_ranking(result, top_n=args.top)

    if args.output:
        result["low"].to_csv(args.output, encoding="utf-8-sig")
        print(f"ランキングをCSV出力: {args.output}")

    logger.info("=== 全工程完了 ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="日本株 翌日高値/安値5%到達確率ランキングシステム")
    parser.add_argument("--debug", action="store_true", help="デバッグモード（外部ライブラリの詳細ログを表示）")
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="全銘柄データを取得")
    p_fetch.add_argument("--refresh", action="store_true", help="キャッシュを無視して再取得")
    p_fetch.set_defaults(func=cmd_fetch)

    # train
    p_train = subparsers.add_parser("train", help="モデルを学習（デフォルトは追加学習）")
    p_train.add_argument("--full", action="store_true", help="フルリトレイン（追加学習でなくゼロから学習）")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = subparsers.add_parser("predict", help="翌日 高値+5%% / 安値-5%% 確率を予測")
    p_predict.add_argument("--top", type=int, default=config.RANKING_TOP_N, help="表示する上位銘柄数")
    p_predict.add_argument("--output", "-o", type=str, default=None, help="CSV出力ファイルパス")
    p_predict.set_defaults(func=cmd_predict)

    # run (ワンショット)
    p_run = subparsers.add_parser("run", help="fetch→追加学習→predict をワンショット実行")
    p_run.add_argument("--top", type=int, default=config.RANKING_TOP_N, help="表示する上位銘柄数")
    p_run.add_argument("--output", "-o", type=str, default=None, help="CSV出力ファイルパス")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    _setup_logging(debug=args.debug)
    args.func(args)


if __name__ == "__main__":
    main()
