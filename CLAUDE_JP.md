# CLAUDE_JP.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを日本語で提供します。

## よく使用するコマンド

### アプリケーションの実行
```bash
# メインのStreamlit Webアプリケーションを起動
streamlit run app.py

# 核となる認識システムをテスト
python enhanced_speaker_recognition.py

# 精度向上のための背景埋め込みを作成
python create_embeddings.py --jvs-path [JVS_PATH] --cv-max-samples 5000
```

### GPUセットアップ (CUDA 12.6)
```bash
# 最初にGPU対応PyTorchをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# その後、残りの依存関係をインストール
pip install -r requirements.txt
```

### 背景データの処理
```bash
# ダウンロードしたJVSデータセットから埋め込みを作成
python prep_jvs_embeddings.py [JVS_PATH] --per-speaker 50

# Common Voice埋め込みを作成（Hugging Face認証が必要）
python prep_common_voice_embeddings.py --max-samples 5000

# テスト用のダミー埋め込みを作成
python create_dummy_cv_embeddings.py
```

### テストと検証
```bash
# データセットマネージャーの機能をテスト
python dataset_manager.py

# 背景ローダーをテスト
python background_loader.py
```

## アーキテクチャ概要

これはSpeechBrainのECAPA-TDNNモデルを使用した日本語話者認識システムです。明確な関心の分離を持つモジュラー設計に従っています：

### 核となるコンポーネント

**メインアプリケーション (`app.py`)**
- セッション状態管理を持つStreamlit Webインターface
- マルチタブUI：話者識別、話者管理、統計情報
- リアルタイムモデル初期化とキャッシュ制御
- UIチェックボックスによる設定管理

**認識エンジン (`enhanced_speaker_recognition.py`)**
- 主要な`JapaneseSpeakerRecognizer`クラス
- 音声前処理、埋め込み抽出、類似度スコアリングを処理
- 背景モデルを使用したAS-Norm（適応スコア正規化）を実装
- キャッシュシステム付き話者データベース管理（`enrolled_speakers_embeddings.npz`）
- ローカル話者とデータセット話者（JVS/Common Voice）の両方をサポート

**データセット管理 (`dataset_manager.py`)**
- `DatasetManager`クラスが話者分類とフィルタリングを処理
- JVS話者（jvs001-jvs100）、Common Voice話者、カスタム話者を区別
- 背景話者の設定可能な除外ルール
- 音声ファイル発見と検証

**背景処理 (`background_loader.py`)**
- `BackgroundEmbeddingLoader`がスコア正規化用の事前計算済み埋め込みを管理
- `.npz`ファイルからJVSとCommon Voice埋め込みを読み込み
- AS-Norm用の結合背景埋め込みを提供
- 埋め込みが利用できない場合は音声ファイル処理に自動フォールバック

### データフロー

1. **初期化**: SpeechBrainモデルを読み込み、`enroll/`フォルダから話者データベースを構築
2. **背景モデル**: スコア正規化用の背景埋め込みを読み込みまたは計算
3. **認識**: 入力音声から埋め込みを抽出し、登録話者とのコサイン類似度を計算
4. **スコアリング**: 背景統計を使用してAS-Normを適用し、信頼度スコア付きの上位マッチを返す

### 設定システム

`config.json`の中央設定が以下を制御：
- モデル設定（デバイス、SpeechBrainモデル）
- 音声処理（サンプリングレート、持続時間制限）
- 認識しきい値と正規化
- データセット包含/除外ルール
- UI表示設定
- キャッシュ動作

### ファイル構造

```
enroll/                          # 話者登録ディレクトリ
├── [話者名]/                    # 個別話者フォルダ
│   └── *.wav                    # 各話者の音声サンプル

background_datasets/             # オプション：背景モデル用生音声
├── jvs/                        # JVSコーパス音声ファイル
└── common_voice_ja/            # Common Voice音声ファイル

background_jvs_ecapa.npz         # 事前計算済みJVS埋め込み（3.8MB）
background_common_voice_ja_ecapa.npz  # 事前計算済みCommon Voice埋め込み（19MB）
enrolled_speakers_embeddings.npz # キャッシュされた話者埋め込み
```

### 主要な設計パターン

**キャッシュ戦略**: システムは高コストな操作に対して積極的なキャッシュを使用：
- 話者埋め込みはタイムスタンプ検証付きで`.npz`ファイルにキャッシュ
- 背景埋め込みはデータセットダウンロードを避けるため事前計算
- 設定変更とファイル変更に基づくキャッシュ無効化

**モジュラー認識パイプライン**: 各ステップが分離：
- 音声前処理（リサンプリング、正規化、持続時間チェック）
- 埋め込み抽出（SpeechBrain ECAPA-TDNN）
- 類似度計算（コサイン類似度）
- スコア正規化（背景統計によるAS-Norm）

**柔軟な話者管理**: システムは複数の話者ソースをサポート：
- `enroll/`フォルダのカスタム話者
- JVSコーパス話者（設定可能な包含）
- Common Voice話者（設定可能な包含）
- 背景話者除外ルール

### 開発ノート

- システムはCUDA、MPS（Apple Silicon）を自動検出、またはCPUにフォールバック
- パフォーマンスのため背景埋め込みが生音声処理より優先
- 話者IDは特定パターンに従う：JVSは`jvs001-jvs100`、Common Voiceは`cv_*`/`commonvoice_*`
- すべての音声は16kHzモノラルに正規化、持続時間制限あり（2-30秒）
- 認識しきい値は設定可能だがデフォルトは0.25
- Streamlitセッション状態がリクエスト間でモデルと話者データベースを保持

### パフォーマンス考慮事項

- GPU加速により埋め込み抽出が3-5倍高速化
- 事前計算背景埋め込みにより大規模データセット保存が不要（2.7GB → 23MB）
- 話者埋め込みキャッシュにより起動時間が数分から数秒に短縮
- 背景モデルサイズは設定可能（デフォルト：100サンプル）でメモリ/精度のトレードオフ