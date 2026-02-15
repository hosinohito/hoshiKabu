# kabu — 日本株 翌日値動き予測ランキングシステム

東証全銘柄の翌日上昇/値下がり確率をLightGBMで予測し、ランキングを出力するシステム。

## 必要環境

- Python 3.10+
- CUDA対応GPU（LightGBM GPU版を使用。CPUのみの場合は`config.py`の`device_type`を`"cpu"`に変更）
- Windows 10/11（開発・検証環境）

## インストール手順

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd kabu
```

### 2. Python仮想環境の作成（推奨）

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

LightGBM GPU版を使う場合は、別途CUDAとLightGBMのGPUビルドが必要です:
```bash
pip install lightgbm --install-option=--gpu
```

### 4. CPU環境の場合

`config.py`の`LIGHTGBM_PARAMS`から以下の3行を削除またはコメントアウト:
```python
"device_type": "gpu",
"gpu_platform_id": 0,
"gpu_device_id": 0,
```

同様に`eval_walkforward.py`の`FAST_PARAMS`からも同じ3行を削除してください。

## 使い方

### 初回セットアップ（データ取得 → フル学習 → 予測）

```bash
# 1. 全銘柄の価格データを取得（約30分〜1時間、銘柄数による）
python main.py fetch

# 2. モデルをゼロからフル学習
python main.py train --full

# 3. 翌日の値動きランキングを表示
python main.py predict
```

### 毎日の運用（ワンライナー）

毎朝 市場開始前、または毎晩 市場終了後に以下を実行:

```bash
python main.py run -o ranking.csv
```

これ1つで **データ差分取得 → 追加学習 → 予測ランキング出力** が完了します。

- 前回学習以降の新しい価格データのみを差分取得
- 既存モデルをベースに追加学習（前回学習日以降のデータで継続学習）
- 値下がり/上昇確率ランキングをコンソールに表示し、`ranking.csv`にも出力

### サブコマンド一覧

| コマンド | 説明 |
|---------|------|
| `python main.py fetch` | 全銘柄の価格データを取得（差分取得） |
| `python main.py fetch --refresh` | キャッシュを無視して全データ再取得 |
| `python main.py train` | 追加学習（既存モデルがなければフル学習） |
| `python main.py train --full` | ゼロからフル学習 |
| `python main.py predict` | 予測ランキングを表示 |
| `python main.py predict --top 20` | 上位20銘柄のみ表示 |
| `python main.py predict -o result.csv` | CSV出力 |
| `python main.py run` | fetch → 追加学習 → predict のワンショット実行 |
| `python main.py run -o ranking.csv` | ワンショット実行 + CSV出力 |

### ウォークフォワード検証

モデル精度をデータリークなしで検証:

```bash
python eval_walkforward.py
```

テスト期間（後ろ15%）で1日ずつ学習→予測を繰り返し、正解率を算出します。

## 出力例

```
======================================================================
  本日の推奨シグナル: 値下がり
  銘柄: 1234.T  サンプル株式会社
  確信度: 65.3%
======================================================================

  翌日値下がり確率ランキング (出来高>=10,000)
======================================================================
順位  コード   銘柄名               業種           値下がり確率
----------------------------------------------------------------------
   1  1234.T   サンプル株式会社       サービス業      65.3%
   2  5678.T   テスト工業            機械            63.1%
  ...
```

## Enhanced モード（実験的）

`config.py` の `ENHANCED_MODE = True` に変更すると、以下の改善を含む拡張パイプラインが有効になります:

1. **新テクニカル指標**: RSI, MACD, Bollinger Bands, 出来高スパイク, セクター相対強度（7特徴量追加）
2. **アンサンブル**: パラメータの異なる3つのLightGBMモデルの予測確率を平均
3. **確率キャリブレーション**: Isotonic regression で確率を補正
4. **非対称閾値**: UP/DOWN方向で異なる確信度閾値（`UP_THRESHOLD`, `DOWN_THRESHOLD`）
5. **サンプル重み付け**: 指数減衰で直近データを重視

```python
# config.py を編集
ENHANCED_MODE = True   # デフォルトは False
```

有効化後、通常通り `python main.py train --full` → `python main.py predict` で Enhanced パイプラインが使用されます。

### Enhanced モードの比較検証

`eval_compare.py` で Baseline と Enhanced を同一条件で公平比較できます:

```bash
python eval_compare.py
```

### 比較検証結果（2025-03 ~ 2026-02, 224営業日）

| 指標 | Baseline | Enhanced | 差分 |
|------|----------|----------|------|
| 統合TOP1 | 72.8% | 73.2% | +0.4pp |
| 上昇のみ | 46.4% | 48.2% | +1.8pp |
| 値下がりのみ | 72.8% | 75.0% | +2.2pp |

値下がり予測で +2.2pp の改善。上昇予測も +1.8pp 改善しているものの、依然50%未満。

## 現在の精度

ウォークフォワード検証（2025-03 ~ 2026-02, 224営業日）:

| 戦略 | 正解率 |
|------|--------|
| 統合戦略 TOP1 | 74.1% |
| 値下がり予測 単独 | 74.1% |
| 上昇予測 単独 | 45.1% |

値下がり予測が主力です。上昇予測の精度改善は今後の課題です。

## パフォーマンス最適化

以下の最適化を実施済みです。

### 第1弾: コア処理の効率化
- **build_dataset二重呼び出し排除**: `cmd_run()`内で学習時に構築したdatasetを予測時にそのまま再利用
- **groupbyオブジェクト使い回し**: `compute_individual_features()`内で同一シンボルへのgroupbyを1回だけ構築し全特徴量計算で共有
- **PCA pivotキャッシュ**: fit時にpivotテーブルをparquet保存し、predict時は差分日付のみ追記して再計算を回避

### 第2弾: I/O・ループの効率化
- **parquet再読込排除**: `cmd_run()`で`fetch_price_data()`/`fetch_index_data()`の戻り値をそのまま使い、同じファイルのディスク再読込（数百MB）を削除
- **pd.to_datetime()統合**: 同一カラムへの重複呼び出しを1変数に集約
- **セクターマッピング最適化**: ネストされた辞書lambdaを平坦辞書の`.map()`に置き換え
- **日次フィルタのgroupby事前辞書化**: 毎日の全行スキャン O(n)×日数 を、事前`groupby("date")`による O(1)辞書参照に改善（eval_walkforward.py, eval_lookback.py）
- **nlargest/nsmallest化**: `sort_values().head()` を `nlargest()`/`nsmallest()` に置き換え、O(n log n) → O(n) ヒープ選択に改善
- **eval_lookback.py groupby使い回し**: `add_lookback_features()`のgroupbyオブジェクトを外部から渡し、10回のルックバックループで再構築を回避

## ファイル構成

```
kabu/
├── config.py            # 設定値（パラメータ、パス等。ENHANCED_MODE切替）
├── main.py              # CLIエントリポイント
├── eval_walkforward.py  # ウォークフォワード検証
├── eval_compare.py      # Baseline vs Enhanced 比較検証
├── eval_lookback.py     # ルックバック期間評価
├── requirements.txt     # 依存パッケージ
├── src/
│   ├── fetcher.py       # データ取得（JPX銘柄リスト、yfinance）
│   ├── preprocessor.py  # 特徴量生成（PCA、市場指数等）
│   ├── model.py         # モデル学習・保存・追加学習
│   └── predictor.py     # 予測・ランキング表示
├── data/                # 価格データキャッシュ（gitignore対象）
└── models/              # 学習済みモデル（gitignore対象）
```
