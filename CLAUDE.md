# CLAUDE.md - プロジェクト開発メモ（現行）

## 概要

東証全銘柄を対象に、翌日の始値基準イベントを2本のLightGBMで予測する。

- `target_high_5pct`: 翌日高値が翌日始値より `+5%` 以上
- `target_low_5pct`: 翌日安値が翌日始値より `-5%` 以上

予測時は上記2確率をそれぞれ全銘柄でランキング表示する。

## 現在のディレクトリ責務

- `main.py`
- CLI (`fetch/train/predict/run`)
- `src/fetcher.py`
- JPX銘柄リスト取得、yfinance取得、キャッシュ
- `src/preprocessor.py`
- 特徴量生成、2ターゲット生成、PCA/市場特徴量統合
- `src/model.py`
- 2ターゲット用2モデル学習、保存/読込
- `src/predictor.py`
- `HIGH_5PCT` / `LOW_5PCT` ランキング生成
- `eval_walkforward.py`
- 新戦略用ウォークフォワード評価

## 直近の仕様変更

1. 旧戦略（翌日始値->終値のUP/DOWN分類）を廃止
2. 2ターゲットの独立分類へ移行
3. モデル保存形式を変更
- `models/lgbm_model.pkl` は `{"models": {"target_high_5pct": ..., "target_low_5pct": ...}, "feature_cols": ...}`
4. 旧形式モデル検出時は `train --full` を要求
5. `predict` / `run` 出力を新ランキング仕様へ変更

## 評価結果（現行設定）

`eval_walkforward.py` 実行条件:

- GPU
- `TRAIN_LOOKBACK_YEARS=3`
- テスト期間: 後ろ10%（`2025-10-31`〜`2026-02-20`, 74営業日）
- リトレイン間隔: 10営業日

結果:

- `HIGH_5PCT`
- TOP1: `40/74 = 54.1%`
- TOP3: `57.7%`
- TOP5: `57.8%`

- `LOW_5PCT`
- TOP1: `52/74 = 70.3%`
- TOP3: `65.8%`
- TOP5: `62.2%`

## 運用メモ

- 通常運用:
- `python main.py run -o ranking.csv`
- 新戦略初回または旧モデル混在時:
- `python main.py train --full`

## 既知の制約

1. `ENHANCED_MODE` は新戦略に未対応（`NotImplementedError`）
2. `eval_compare.py`, `eval_lookback.py` は旧ターゲット前提のため未更新
3. yfinanceの欠損銘柄は再取得ロジックを入れているが、ソース側未提供銘柄は取得不可

## 技術的補足

- `LabelEncoder` は保存・再利用（推論時ずれ防止）
- 特徴量不足列は学習/推論時に0埋め
- 推論入力のNaNは0埋め
- 価格取得は `threads=False` + 外側並列（`max_workers=3`）
