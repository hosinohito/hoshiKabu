# hoshiKabu

日本株の翌営業日における以下2イベントを、LightGBMで銘柄ごとに予測してランキング化するプロジェクトです。

- `HIGH_5PCT`: 翌営業日の高値が始値比 `+5%` 以上に到達
- `LOW_5PCT`: 翌営業日の安値が始値比 `-5%` 以下に到達

## 環境要件

- Windows 10/11 + PowerShell
- Python 3.10+
- GPU (LightGBM GPU を使う設定。CPUで動かしたい場合は `config.py` の `device_type` 変更が必要)

## セットアップ (PowerShell)

このリポジトリ直下 (`hoshiKabu`) で実行してください。

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` には `.xls` 読み込み用の `xlrd` を含んでいます。

## 使い方

### 一括実行

```powershell
python main.py run -o ranking.csv
```

`run` は `fetch -> train -> predict` を順番に実行します。

### 個別実行

```powershell
python main.py fetch
python main.py train --full
python main.py predict -o ranking.csv
```

## 出力

- コンソールに `LOW_5PCT` / `HIGH_5PCT` のランキングを表示
- `-o ranking.csv` 指定時は `LOW_5PCT` ランキングをCSV保存

## 主な構成

- `main.py`: CLIエントリ (`fetch/train/predict/run`)
- `src/fetcher.py`: JPX銘柄リスト・価格データ取得
- `src/preprocessor.py`: 特徴量生成、ターゲット生成
- `src/model.py`: 2ターゲット学習、保存、読み込み
- `src/predictor.py`: 全銘柄推論、ランキング作成
- `config.py`: 学習・推論・データ取得設定

## 評価スクリプト

```powershell
python eval_walkforward.py
python eval_lookback_top1.py
```

## 注意事項

- `ENHANCED_MODE` は未対応です。`config.py` では `ENHANCED_MODE = False` を使用してください。
- 既存モデルとの特徴量不一致がある場合、`python main.py train --full` で再学習してください。
- `eval_compare.py` と `eval_lookback.py` は旧評価コードのため、現在の主タスク評価には `eval_walkforward.py` / `eval_lookback_top1.py` を優先してください。

## トラブルシュート

### 1. `py` コマンドが見つからない

`py` ではなく `python` を使ってください。

```powershell
python --version
where.exe python
```

### 2. `Activate.ps1` 実行時に実行ポリシーエラー

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. `No module named 'xlrd'`

```powershell
pip install -r requirements.txt
```

### 4. activate せずに実行したい

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py run -o ranking.csv
```
