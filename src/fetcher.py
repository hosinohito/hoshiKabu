"""JPX銘柄リスト取得・yfinanceバッチダウンロード・Parquetキャッシュ"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def fetch_stock_list(use_cache: bool = True) -> pd.DataFrame:
    """JPX公式Excelから東証全上場銘柄リストを取得する。"""
    cache_path = config.DATA_RAW_DIR / config.STOCK_LIST_CACHE
    if use_cache and cache_path.exists():
        logger.info("銘柄リストをキャッシュから読み込み")
        return pd.read_parquet(cache_path)

    logger.info("JPXから銘柄リストをダウンロード中...")
    resp = requests.get(config.JPX_LIST_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_excel(BytesIO(resp.content), dtype=str)

    # カラム名を位置ベースで正規化（JPX Excelの列順序は固定）
    # 0:日付, 1:コード, 2:銘柄名, 3:市場・商品区分, 4:33業種コード, 5:33業種区分, ...
    orig_cols = list(df.columns)
    col_map = {}
    if len(orig_cols) >= 3:
        col_map[orig_cols[1]] = "code"
        col_map[orig_cols[2]] = "name"
    if len(orig_cols) >= 4:
        col_map[orig_cols[3]] = "market"
    if len(orig_cols) >= 6:
        col_map[orig_cols[4]] = "sector_code"
        col_map[orig_cols[5]] = "sector_name"
    df = df.rename(columns=col_map)

    required = ["code", "name"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"必須カラム '{col}' が見つかりません。カラム一覧: {list(df.columns)}")

    # コードが数字または英数字混合（285A等）の行のみ残す
    df = df[df["code"].astype(str).str.match(r"^[0-9A-Za-z]+$", na=False)].copy()
    df["symbol"] = df["code"].astype(str) + ".T"

    # セクター情報が無い場合のフォールバック
    if "sector_code" not in df.columns:
        df["sector_code"] = "0"
    if "sector_name" not in df.columns:
        df["sector_name"] = "不明"

    keep_cols = ["code", "symbol", "name", "sector_code", "sector_name"]
    if "market" in df.columns:
        keep_cols.append("market")
    df = df[keep_cols].reset_index(drop=True)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"銘柄リスト取得完了: {len(df)}銘柄")
    return df


def _download_batch(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """yfinanceで複数銘柄を一括ダウンロードし、リトライ付きで返す。"""
    for attempt in range(config.YFINANCE_MAX_RETRIES):
        try:
            data = yf.download(
                symbols,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            return data
        except Exception as e:
            logger.warning(f"ダウンロード失敗 (試行{attempt + 1}): {e}")
            time.sleep(config.YFINANCE_SLEEP_SEC * (attempt + 1))
    logger.error(f"ダウンロード失敗（リトライ上限）: {symbols[:3]}...")
    return pd.DataFrame()


def _parse_batch_data(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """バッチダウンロード結果をlong形式に変換する。"""
    records = []
    if raw.empty:
        return pd.DataFrame()

    if len(symbols) == 1:
        sym = symbols[0]
        df = raw.copy()
        df["symbol"] = sym
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        records.append(df)
    else:
        for sym in symbols:
            try:
                if sym in raw.columns.get_level_values(0):
                    df = raw[sym].copy()
                    df = df.dropna(how="all")
                    if df.empty:
                        continue
                    df["symbol"] = sym
                    df = df.reset_index()
                    if "Date" in df.columns:
                        df = df.rename(columns={"Date": "date"})
                    records.append(df)
            except (KeyError, TypeError):
                continue

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    # カラム名を小文字に統一
    result.columns = [c.lower() if isinstance(c, str) else c for c in result.columns]
    return result


def fetch_price_data(
    stock_list: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """全銘柄の価格データをバッチ取得し、Parquetキャッシュに保存する。"""
    cache_path = config.DATA_RAW_DIR / "prices.parquet"
    start_date = start_date or config.DATA_START_DATE

    # キャッシュから差分取得
    existing = None
    if use_cache and cache_path.exists():
        existing = pd.read_parquet(cache_path)
        existing["date"] = pd.to_datetime(existing["date"])
        last_date = existing["date"].max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"キャッシュあり: {last_date.date()} まで取得済み。{start_date}から差分取得")

    symbols = stock_list["symbol"].tolist()
    all_data = []
    batches = []
    for i in range(0, len(symbols), config.YFINANCE_BATCH_SIZE):
        batches.append(symbols[i : i + config.YFINANCE_BATCH_SIZE])
    total_batches = len(batches)

    max_workers = 3

    def _fetch_one(batch_idx: int, batch: list[str]) -> pd.DataFrame:
        logger.info(f"バッチ {batch_idx+1}/{total_batches} ({len(batch)}銘柄) ダウンロード中...")
        raw = _download_batch(batch, start=start_date, end=end_date)
        return _parse_batch_data(raw, batch)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            parsed = future.result()
            if not parsed.empty:
                all_data.append(parsed)

    if all_data:
        new_data = pd.concat(all_data, ignore_index=True)
        if existing is not None:
            new_data["date"] = pd.to_datetime(new_data["date"])
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        else:
            combined = new_data
    elif existing is not None:
        combined = existing
        logger.info("新規データなし。キャッシュをそのまま使用")
    else:
        logger.warning("価格データを取得できませんでした")
        return pd.DataFrame()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    logger.info(f"価格データ保存完了: {len(combined)}行, {combined['symbol'].nunique()}銘柄")
    return combined


def fetch_index_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """日経225・TOPIX(ETF代替)の指数データを取得する。"""
    cache_path = config.DATA_RAW_DIR / "index_prices.parquet"
    start_date = start_date or config.DATA_START_DATE
    symbols = [config.NIKKEI225_SYMBOL, config.TOPIX_SYMBOL]

    logger.info("市場指数データをダウンロード中...")
    raw = _download_batch(symbols, start=start_date, end=end_date)
    parsed = _parse_batch_data(raw, symbols)

    if parsed.empty:
        if cache_path.exists():
            logger.info("指数データのキャッシュを使用")
            return pd.read_parquet(cache_path)
        return pd.DataFrame()

    parsed["date"] = pd.to_datetime(parsed["date"])
    parsed.to_parquet(cache_path, index=False)
    logger.info(f"指数データ保存完了: {len(parsed)}行")
    return parsed
