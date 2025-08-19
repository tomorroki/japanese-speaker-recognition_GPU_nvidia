# CLAUDE_JP.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを日本語で提供します。

## システム概要

これはSpeechBrainのECAPA-TDNNモデルとpyannote.audioを使用した日本語話者認識・複数話者分析システムです。単一話者識別、包括的な複数話者ダイアライゼーションによる時系列分析、そしてstreamlit-advanced-audio (audix) による真のリアルタイム音声同期を使った手動話者分離機能を提供します。

## よく使用するコマンド

### アプリケーションの実行
```bash
# メインのStreamlit Webアプリケーションを起動
streamlit run app.py

# 核となる認識システムをテスト（単一話者）
python enhanced_speaker_recognition.py

# 複数話者分析システムをテスト
python speaker_diarization.py

# 精度向上のための背景埋め込みを作成
python create_embeddings.py --jvs-path [JVS_PATH] --cv-max-samples 5000
```

### GPUセットアップ (CUDA 12.6)
```bash
# 最初にGPU対応PyTorchをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# その後、残りの依存関係をインストール（streamlit-advanced-audio含む）
pip install -r requirements.txt
```

### 複数話者分析セットアップ
```bash
# pyannote.audio用のHugging Face tokenを設定（必須）
echo "HF_TOKEN=your_huggingface_token_here" > .env

# 利用規約同意: https://huggingface.co/pyannote/speaker-diarization
```

### 背景データの処理
```bash
# ダウンロードしたJVSデータセットから埋め込みを作成
python prep_jvs_embeddings.py [JVS_PATH] --per-speaker 50

# Common Voice埋め込みを作成（Hugging Face認証が必要）
python prep_common_voice_embeddings.py --max-samples 5000

# テスト用のダミー埋め込みを作成（クイックスタート）
python create_dummy_cv_embeddings.py
```

### テストと検証
```bash
# データセットマネージャーの機能をテスト
python dataset_manager.py

# 背景ローダーをテスト
python background_loader.py

# セグメントプロセッサーをテスト（複数話者）
python segment_processor.py

# 話者ダイアライゼーションをテスト
python speaker_diarization.py
```

## アーキテクチャ概要

このシステムは、モジュラーな2段階アプローチを使用して単一話者認識と複数話者ダイアライゼーション分析を組み合わせています：

### 核となるコンポーネント

**メインアプリケーション (`app.py`)**
- 4タブ構造のStreamlit Webインターフェース
- 単一話者・複数話者システム両方のセッション状態管理
- リアルタイムモデル初期化とキャッシュ制御
- UI制御による設定管理
- タブ: 🎤 単一話者, 🎭 複数話者分析, 👥 話者管理, 📊 統計情報

**単一話者認識 (`enhanced_speaker_recognition.py`)**
- 主要な`JapaneseSpeakerRecognizer`クラス
- ECAPA-TDNN埋め込み抽出と類似度スコアリング
- 背景モデルを使用したAS-Norm（適応スコア正規化）
- キャッシュ付き話者データベース管理（`enrolled_speakers_embeddings.npz`）
- ローカル話者とデータセット話者（JVS/Common Voice）の両方をサポート
- 複数話者統合用の新しい`recognize_segment()`メソッド

**複数話者分析システム (`speaker_diarization.py`)**
- 時系列話者分離用pyannote.audioを使用する`SpeakerDiarizer`クラス
- ダイアライゼーションと認識を統合する`MultiSpeakerRecognizer`クラス
- 2段階処理：ダイアライゼーション → セグメント認識
- タイムラインデータと信頼度スコアを含む包括的結果構造
- 設定可能な話者数とフィルタリングオプションのサポート

**セグメント処理 (`segment_processor.py`)**
- 音声セグメント抽出を処理する`SegmentProcessor`クラス
- 認識用音声前処理（16kHz変換、正規化）
- 生ダイアライゼーション結果と認識システム間の橋渡し
- 音声セグメントの検証と品質管理

**データセット管理 (`dataset_manager.py`)**
- 話者分類機能を持つ拡張された`DatasetManager`クラス
- JVS話者（jvs001-jvs100）、Common Voice、カスタム話者を区別
- 背景話者の設定可能な除外ルール
- 音声ファイル発見と検証

**背景処理 (`background_loader.py`)**
- 事前計算済み埋め込みを管理する`BackgroundEmbeddingLoader`
- `.npz`ファイルからJVSとCommon Voice埋め込みを読み込み
- AS-Norm用の結合背景埋め込みを提供
- 埋め込みが利用できない場合は音声処理に自動フォールバック

### 複数話者分析アーキテクチャ

#### 2段階処理パイプライン
1. **ダイアライゼーション段階**: pyannote.audioが生音声を処理して「いつ誰が話しているか」を検出
2. **認識段階**: 検出された各セグメントをECAPA-TDNNで話者識別処理

#### データフロー
1. **音声入力**: Streamlit経由で複数話者音声ファイルをアップロード
2. **ダイアライゼーション**: pyannote.audioが時系列話者セグメントを作成
3. **セグメント抽出**: 各セグメントを抽出し16kHzに前処理
4. **認識**: 各セグメントを登録話者に対して認識
5. **統合**: 結果を包括的なタイムライン分析に統合
6. **可視化**: 複数のチャートタイプ（ダイアライゼーションタイムライン、話者タイムライン、統計）

### 設定システム

`config.json`の中央設定がすべての側面を制御：

```json
{
  "model": {
    "speechbrain_model": "speechbrain/spkrec-ecapa-voxceleb",
    "device": "auto"
  },
  "audio": {
    "sample_rate": 16000,
    "duration_min": 2.0,
    "duration_max": 30.0
  },
  "recognition": {
    "threshold": 0.25,
    "use_score_normalization": true,
    "background_speakers_count": 100
  },
  "diarization": {
    "model": "pyannote/speaker-diarization-3.1",
    "min_speakers": 1,
    "max_speakers": 10,
    "min_segment_duration": 0.5
  },
  "segment_processing": {
    "target_sample_rate": 16000,
    "normalize": true
  },
  "datasets": {
    "exclude_background_speakers": true,
    "allow_jvs_speakers": true,
    "allow_common_voice_speakers": false
  },
  "ui": {
    "show_jvs_in_results": false,
    "show_common_voice_in_results": false
  }
}
```

### ファイル構造

```
japanese-speaker-recognition_GPU_nvidia/
├── app.py                                    # メインStreamlitアプリケーション（4タブ）
├── enhanced_speaker_recognition.py           # 単一話者認識システム
├── speaker_diarization.py                   # 複数話者分析システム（NEW）
├── segment_processor.py                     # 音声セグメント処理（NEW）
├── dataset_manager.py                       # データセットと話者管理
├── background_loader.py                     # 背景埋め込み管理
├── config.json                              # 包括的設定
├── .env                                     # 環境変数（HF_TOKEN）
├── requirements.txt                         # 依存関係（pyannote.audio含む）

# 埋め込みファイル
├── background_jvs_ecapa.npz                 # JVS背景埋め込み（3.8MB）
├── background_common_voice_ja_ecapa.npz     # Common Voice埋め込み（19MB）
├── enrolled_speakers_embeddings.npz         # 話者データベースキャッシュ

# 話者登録
├── enroll/                                  # ユーザー話者登録
│   ├── speaker1/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── speaker2/

# 処理スクリプト
├── prep_jvs_embeddings.py                   # JVS埋め込み作成
├── prep_common_voice_embeddings.py          # Common Voice埋め込み作成
├── create_embeddings.py                     # 統合埋め込み作成
├── create_dummy_cv_embeddings.py            # クイックスタート用ダミーデータ

# ドキュメント
├── README.md                                # 完全なシステムドキュメント
├── QUICK_START_GUIDE.md                     # 両モード用クイックスタート
├── SPEAKER_MANAGEMENT_GUIDE.md              # 話者管理ガイド
├── FAQ.md                                   # 包括的FAQ
└── CLAUDE_JP.md                             # このファイル
```

### 主要な設計パターン

**モジュラー2段階アーキテクチャ**: 
- ダイアライゼーション（pyannote.audio）と認識（ECAPA-TDNN）間の明確な分離
- 各段階が適切な前処理でその特定タスクに最適化
- セグメントプロセッサーが段階間のギャップを橋渡し

**統合話者管理**: 
- 単一登録システム（`enroll/`フォルダ）が単一・複数話者モード両方に対応
- すべての機能にわたって一貫した話者データベース
- JVS/Common Voice話者用の柔軟なフィルタリングオプション

**包括的キャッシュ戦略**: 
- タイムスタンプ検証付き話者埋め込みキャッシュ
- 大規模データセットダウンロードを避ける事前計算背景埋め込み
- 最適パフォーマンスのための多レベルキャッシュ

**インタラクティブ可視化**:
- 複数タイムラインビュー：ダイアライゼーション結果と話者認識結果
- リアルタイムフィルタリングと設定オプション
- トップ5候補スコア付き詳細セグメント別分析

### 処理要件

#### 単一話者認識
- 入力: 2-30秒、単一話者音声
- 処理: ECAPA-TDNN埋め込み + AS-Normスコアリング
- 出力: 信頼度スコア付き話者識別

#### 複数話者分析
- 入力: 複数話者音声（任意長、2-10話者）
- 処理: pyannote.audioダイアライゼーション → セグメント認識
- 出力: 話者切り替えパターンのタイムライン分析

### 環境変数

```bash
# 複数話者分析に必須
HF_TOKEN=your_huggingface_token_here
```

### 依存関係

複数話者機能の主要依存関係:
```
pyannote.audio>=3.1.0
python-dotenv>=0.19.0
speechbrain>=0.5.12
torch>=1.13.0
streamlit>=1.20.0
plotly>=5.0.0
```

## 開発ノート

### 複数話者分析実装

- **pyannote.audio**: 自動話者検出によるダイアライゼーション処理
- **セグメント認識**: 各ダイアライゼーションセグメントを既存認識パイプラインで個別処理
- **結果統合**: 時系列と認識情報を組み合わせた包括的データ構造
- **可視化**: タイムライン分析用Plotlyベースインタラクティブチャート

### 話者データベース統合

- **統合登録**: 同じ`enroll/`フォルダ構造が両モードに対応
- **自動検出**: システムがCUDA/MPS/CPUを自動検出
- **背景埋め込み**: AS-Normが単一・複数話者認識両方で動作
- **フィルタリングオプション**: JVS/Common Voice話者を独立して表示/非表示

### 設定管理

- **単一設定ファイル**: `config.json`がすべてのシステム側面を制御
- **UI統合**: Streamlitインターフェース経由で多くの設定を制御可能
- **環境変数**: 機密データ（HF_TOKEN）は`.env`ファイル
- **ランタイム設定**: 再起動なしで設定変更可能

### パフォーマンス考慮事項

- **GPU加速**: ダイアライゼーションと認識両方で3-5倍高速化
- **背景埋め込み**: 2.7GB+データセット保存の必要性を排除
- **セグメントキャッシュ**: 類似音声セグメントの重複処理を削減
- **メモリ管理**: 設定可能な背景話者数と最大話者数

### 認証要件

複数話者分析に必要:
1. Hugging Faceアカウントとトークン
2. pyannote/speaker-diarization利用規約同意
3. `.env`ファイルでのトークン設定

これによりモデルライセンスと使用規約への準拠を保証。

## 重要な実装詳細

### セグメント処理フロー

1. **ダイアライゼーション**: 生音声 → pyannote.audio → 時系列セグメント
2. **抽出**: 元音声からセグメント抽出
3. **前処理**: 認識用にセグメントを16kHzに変換・正規化
4. **認識**: 各セグメントをECAPA-TDNNパイプラインで処理
5. **統合**: 結果を時系列情報と統合

### 話者ID パターン

- **カスタム話者**: `enroll/`フォルダ内の任意の名前
- **JVS話者**: `jvs001`から`jvs100`
- **Common Voice話者**: `cv_*`または`commonvoice_*`接頭辞

### 可視化機能

- **ダイアライゼーションタイムライン**: SPEAKER_XXラベル付き生pyannote.audio結果
- **話者タイムライン**: 実際の話者名による認識ベースタイムライン
- **統計テーブル**: 時間・セグメント・信頼度による話者別分析
- **トップ5表示**: 各セグメントの候補ランキング

## よくある問題のトラブルシューティング

### 複数話者分析の問題

1. **HF_TOKENエラー**: `.env`でトークンが設定され利用規約に同意済みか確認
2. **セグメント検出なし**: 音質、話者分離、持続時間を確認
3. **認識精度悪化**: 話者登録を確認、背景埋め込みを検討
4. **メモリ問題**: 設定で`max_speakers`と`background_speakers_count`を削減

### 一般的な問題

1. **モデル読み込み**: 初回ダウンロード用のインターネット接続を確認
2. **音声形式**: サポート形式（WAV, MP3, FLAC, M4A, OGG）を使用
3. **持続時間制限**: 単一話者は2-30秒、複数話者は制限なし
4. **話者登録**: `enroll/`フォルダに話者あたり2-5音声サンプルを確保

システムは、プロフェッショナルグレードの話者分析機能を提供しながら、堅牢でユーザーフレンドリーになるよう設計されています。