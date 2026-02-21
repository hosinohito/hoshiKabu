# kabu - 日本株 翌日5%到達確率ランキングシステム

東証全銘柄を対象に、翌日の始値を基準として次の2確率をLightGBMで予測しランキング出力します。

- `HIGH_5PCT`: 翌日高値が始値より `+5%` 以上高い確率
- `LOW_5PCT`: 翌日安値が始値より `-5%` 以上低い確率

## 必要環境

- Python 3.10+
- CUDA対応GPU（LightGBM GPU利用）
- Windows 10/11（開発・検証環境）

## セットアップ

```bash
git clone <repository-url>
cd kabu
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### 初回

```bash
python main.py fetch
python main.py train --full
python main.py predict -o ranking.csv
```

### 日次運用

```bash
python main.py run -o ranking.csv
```

`run` は `fetch -> train -> predict` を一括実行します。

## 出力

コンソールには次を表示します。

- 注目シグナル（`HIGH_5PCT` 上位1銘柄、`LOW_5PCT` 上位1銘柄）
- `LOW_5PCT` ランキング
- `HIGH_5PCT` ランキング

`-o ranking.csv` 指定時は `LOW_5PCT` ランキングをCSV保存します。

## 現在の評価結果

`eval_walkforward.py`（GPU, `TRAIN_LOOKBACK_YEARS=3`, テスト後ろ10%, 2025-10-31〜2026-02-20, 74営業日）

- `HIGH_5PCT`
- TOP1: `40/74 = 54.1%`
- TOP3: `57.7%`
- TOP5: `57.8%`

- `LOW_5PCT`
- TOP1: `52/74 = 70.3%`
- TOP3: `65.8%`
- TOP5: `62.2%`

### ルックバック比較（TOP1専用）

`eval_lookback_top1.py` で `lookback=1..N (N=1..10)` を比較した結果:

- `BEST_HIGH`: `lookback=1..9`, TOP1 `59.8%`
- `BEST_LOW`: `lookback=1..9`, TOP1 `71.9%`
- `BEST_AVG`: `lookback=1..9`, TOP1 `65.8%`

ただし運用判断としては、過去設定との整合と安定運用を優先し、現時点では `LOOKBACK_DAYS=1..6` を維持する。

## 重要な注意点

- モデル保存形式は新戦略向けに変更済みです。旧モデルがある場合は最初に `python main.py train --full` を実行してください。
- `ENHANCED_MODE` は新戦略に未対応です。`config.py` で `ENHANCED_MODE = False` を使用してください。
- `eval_compare.py` と `eval_lookback.py` は旧ターゲット前提のため、現時点では新戦略評価には使いません。

## 検証コマンド

```bash
python eval_walkforward.py
python eval_lookback_top1.py
```

## 主な実装ファイル

- `src/preprocessor.py`
- `target_high_5pct` / `target_low_5pct` 生成
- `src/model.py`
- 2モデル学習・保存（`target_high_5pct`, `target_low_5pct`）
- `src/predictor.py`
- 2確率ランキング出力
- `main.py`
- CLI運用（`fetch/train/predict/run`）
