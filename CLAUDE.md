# CLAUDE.md — プロジェクト開発メモ

## プロジェクト概要
日本株（東証全銘柄）の翌日値動きを予測するLightGBMベースのランキングシステム。
JPX銘柄リスト取得 → yfinanceで価格取得 → 特徴量生成(PCA市場ファクター含む) → LightGBM学習 → 翌日上昇/値下がり確率ランキング出力。

## ディレクトリ構成
```
kabu/
├── config.py            # 全設定値（パス、パラメータ、ENHANCED_MODE切替）
├── main.py              # CLI（fetch / train / predict / run）
├── eval_walkforward.py  # ウォークフォワード検証スクリプト
├── eval_compare.py      # Baseline vs Enhanced 比較検証スクリプト
├── eval_lookback.py     # ルックバック期間の評価スクリプト
├── requirements.txt
├── src/
│   ├── fetcher.py       # JPX銘柄リスト・yfinance価格取得（英数字コード対応）
│   ├── preprocessor.py  # 特徴量生成（個別銘柄・PCA・市場指数）
│   ├── model.py         # LightGBM学習・保存・追加学習
│   └── predictor.py     # 全銘柄予測・ランキング表示
├── data/
│   ├── raw/             # prices.parquet, index_prices.parquet, stock_list.parquet
│   └── processed/       # pca_model.pkl, label_encoder.pkl
├── models/              # lgbm_model.pkl, train_meta.json
└── ranking.csv          # 直近の予測結果出力
```

## アーキテクチャ
- **ターゲット**: 翌日 始値→終値 が上昇なら1（UP）、下落なら0（DOWN）
- **特徴量**: 個別銘柄リターン系(MA, ボラティリティ, 乖離率) + 出来高系 + PCA市場ファクター(50成分) + 市場指数(日経225/TOPIX) + セクター + 曜日/月
- **モデル**: LightGBM binary classification（GPU対応）
- **追加学習**: 既存モデルをinit_modelとして、前回学習日以降のデータで継続学習
- **予測**: 上昇確率と値下がり確率(=1-上昇確率)を比較し、確信度の高い方向の銘柄を推奨

## 現在の精度（ウォークフォワード検証 2025-03 ~ 2026-02）
- **統合戦略 TOP1**: 74.1%（224日中166日正解）
- **値下がり予測 単独**: 74.1%（主力。ほぼ毎日DOWN方向が選択される）
- **上昇予測 単独**: 45.1%（ランダム以下。改善が今後の課題）
- **UP/DOWN独立 TOP-N**:
  - 上昇: TOP1=45.1%, TOP3=46.4%, TOP5=47.0%
  - 値下がり: TOP1=74.1%, TOP3=68.6%, TOP5=67.5%

## Enhanced モード（実験的）
`config.py` の `ENHANCED_MODE = True` で以下の拡張パイプラインが有効になる:
1. **新テクニカル指標 (7列追加)**: RSI, MACD histogram/normalized, Bollinger position/width, volume_spike_ratio, sector_relative_strength
2. **アンサンブル (3モデル)**: 標準/浅く速く/深く正則化 の3パラメータセットLightGBMの予測確率平均
3. **確率キャリブレーション**: Isotonic regression でバリデーション確率を補正
4. **非対称閾値**: `UP_THRESHOLD=0.52`, `DOWN_THRESHOLD=0.55` で方向別に確信度閾値を設定
5. **サンプル重み付け**: `SAMPLE_WEIGHT_DECAY=0.998` の指数減衰で直近データを重視

### 比較検証結果 (eval_compare.py, 2025-03 ~ 2026-02, 224日)
| 指標 | Baseline | Enhanced | 差分 |
|------|----------|----------|------|
| 統合TOP1 | 72.8% | 73.2% | +0.4pp |
| 上昇のみ | 46.4% | 48.2% | +1.8pp |
| 値下がりのみ | 72.8% | 75.0% | +2.2pp |

値下がり単独で +2.2pp 改善。全体は +0.4pp の微改善。一部月(6月, 1月)で悪化傾向あり。

### Enhanced 関連の実装箇所
- `config.py`: `ENHANCED_MODE`, テクニカル指標パラメータ, アンサンブルパラメータ, 閾値, 重み減衰率
- `src/preprocessor.py`: `compute_individual_features()` に RSI/MACD/BB/volume_spike 追加, `build_dataset()` に sector_relative_strength 追加
- `src/model.py`: `compute_sample_weights()`, `train_ensemble()`, `ensemble_predict_proba()`, `fit_calibrator()`, `save_model_enhanced()`, `load_model_enhanced()`
- `src/predictor.py`: `predict_all_enhanced()`
- `main.py`: `_train_enhanced()`, cmd_train/cmd_predict/cmd_run で ENHANCED_MODE 分岐

## 既知の課題・今後の改善候補
1. **上昇予測の精度が低い**: ターゲット不均衡(DOWN 56.77% vs UP 43.23%)が根本原因。`is_unbalance: True`を追加済みだが効果限定的。確信度比較で依然としてDOWN側が支配的
2. **改善アプローチ候補**:
   - UP/DOWN別に独立したモデルを学習する
   - ターゲット定義の変更（始値→終値ではなく、前日終値→当日終値など）
   - UP予測専用の特徴量エンジニアリング
   - Enhanced モードの閾値チューニング（現状 UP=0.52, DOWN=0.55）
3. **GPU依存**: config.pyの`device_type: "gpu"`が前提。CPU環境では変更が必要

## 技術的な注意点
- 銘柄コードは英数字混合に対応（285A, 130A等。2024年以降の東証新コード体系）
- yfinanceのバッチ取得はスリープ付きリトライ（レート制限対策）
- PCAモデルは学習時にfit、予測時はtransformのみ（列数不一致時はパディング）
- 追加学習は`lgb.LGBMClassifier.fit(init_model=既存モデル)`で実現
- eval_walkforward.pyは独自のFAST_PARAMSを持つ（検証高速化のため）
- 出来高フィルタ（デフォルト10,000）で低流動性銘柄を除外

## 実施済みパフォーマンス最適化

### 第1弾
- build_dataset二重呼び出し排除（cmd_run内でdataset変数を再利用）
- compute_individual_features内のgroupbyオブジェクト事前作成・使い回し
- PCA pivotテーブルのparquetキャッシュ（fit時に保存、predict時は差分追記のみ）

### 第2弾
- **cmd_run()のparquet再読込排除**: fetch_index_data()/fetch_price_data()の戻り値をそのまま使い、_load_prices()による数百MBの再読込を削除
- **pd.to_datetime()統合**: 同一カラムへの重複呼び出しを1回に（preprocessor.py, eval_lookback.py）
- **セクターマッピング最適化**: ネストされた辞書lambdaを事前展開した平坦辞書の.map()に置き換え（preprocessor.py, eval_lookback.py）
- **eval_walkforward.py日次ループ高速化**: 日次フィルタをgroupby事前辞書化でO(1)参照に、sort_values().head()をnlargest()/nsmallest()に置き換え
- **eval_lookback.py groupby最適化**: add_lookback_features()のgroupbyオブジェクトを外部から渡して10回のループで使い回し、evaluate_model()の日次ループも事前groupby辞書+nlargest()に置き換え
