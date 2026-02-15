# CLAUDE.md — プロジェクト開発メモ

## プロジェクト概要
日本株（東証全銘柄）の翌日値動きを予測するLightGBMベースのランキングシステム。
JPX銘柄リスト取得 → yfinanceで価格取得 → 特徴量生成(PCA市場ファクター含む) → LightGBM学習 → 翌日上昇/値下がり確率ランキング出力。

## ディレクトリ構成
```
kabu/
├── config.py            # 全設定値（パス、パラメータ等）
├── main.py              # CLI（fetch / train / predict / run）
├── eval_walkforward.py  # ウォークフォワード検証スクリプト
├── eval_lookback.py     # ルックバック期間の評価スクリプト
├── requirements.txt
├── src/
│   ├── fetcher.py       # JPX銘柄リスト・yfinance価格取得
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

## 既知の課題・今後の改善候補
1. **上昇予測の精度が低い**: ターゲット不均衡(DOWN 56.77% vs UP 43.23%)が根本原因。`is_unbalance: True`を追加済みだが効果限定的。確信度比較で依然としてDOWN側が支配的
2. **改善アプローチ候補**:
   - UP/DOWN別に独立したモデルを学習する
   - 確率のキャリブレーション（Platt scaling等）を適用する
   - 確信度閾値を方向別に設定する（例: UP > 0.55, DOWN > 0.60 のように非対称閾値）
   - ターゲット定義の変更（始値→終値ではなく、前日終値→当日終値など）
   - UP予測専用の特徴量エンジニアリング
3. **GPU依存**: config.pyの`device_type: "gpu"`が前提。CPU環境では変更が必要

## 技術的な注意点
- yfinanceのバッチ取得はスリープ付きリトライ（レート制限対策）
- PCAモデルは学習時にfit、予測時はtransformのみ（列数不一致時はパディング）
- 追加学習は`lgb.LGBMClassifier.fit(init_model=既存モデル)`で実現
- eval_walkforward.pyは独自のFAST_PARAMSを持つ（検証高速化のため）
- 出来高フィルタ（デフォルト10,000）で低流動性銘柄を除外
