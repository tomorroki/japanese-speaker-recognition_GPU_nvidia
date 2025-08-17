# 日本語話者認識システム

SpeechBrain + ECAPA-TDNNとpyannote.audioを使用した高精度な日本語話者識別・分析Webアプリケーション

## 🎯 概要

このシステムは、**SpeechBrain**の**ECAPA-TDNN**モデルと**pyannote.audio**を組み合わせて、音声ファイルから話者を高精度で識別・分析するWebアプリケーションです。単一話者の識別に加え、複数話者が含まれる音声の時系列分析にも対応しています。

## ✨ 主な機能

### 🎤 単一話者識別
- **高精度認識**: ECAPA-TDNNによる最先端の話者埋め込み技術
- **AS-Norm対応**: 背景モデルを使用したスコア正規化で精度向上
- **トップ10スコア表示**: 全候補話者のスコア詳細分析

### 🎭 複数話者分析（NEW!）
- **話者ダイアライゼーション**: pyannote.audioによる「いつ誰が話しているか」検出
- **時系列話者認識**: 各時間帯の話者を自動識別
- **インタラクティブ可視化**: タイムラインチャートとPlotlyグラフ
- **トップ5表示**: 各セグメントの認識候補をランキング表示

### 📊 可視化・分析機能
- **ダイアライゼーションタイムライン**: pyannote.audioの分離結果
- **話者別タイムライン**: 認識結果ベースの発話パターン
- **話者統計**: セグメント数・発話時間・信頼度の詳細分析
- **フィルタリング**: JVS/Common Voice話者の表示・非表示制御

### 🛠️ システム機能
- **💾 軽量配布**: 音声ファイルではなく埋め込みベクトル（.npz）のみを同梱
- **🖥️ 直感的UI**: Streamlitによる使いやすいWebインターフェース
- **⚙️ 柔軟な設定**: JSON設定ファイルによるパラメータ調整
- **🗂️ 背景話者除外**: JVS・Common Voice話者の自動除外機能

## 🛠️ 技術スタック

- **Python 3.8+**
- **SpeechBrain** (speechbrain/spkrec-ecapa-voxceleb)
- **pyannote.audio** (pyannote/speaker-diarization-3.1)
- **Streamlit** (Web UI)
- **PyTorch** (深層学習フレームワーク)
- **librosa, soundfile** (音声処理)
- **scikit-learn** (機械学習)
- **Plotly** (可視化)
- **Hugging Face** (モデル・データセット)

## 📁 プロジェクト構造

```
japanese-speaker-recognition_GPU_nvidia/
├── app.py                                    # Streamlit メインアプリ
├── enhanced_speaker_recognition.py           # 単一話者認識システム
├── speaker_diarization.py                   # 複数話者分析システム（NEW）
├── segment_processor.py                     # セグメント処理（NEW）
├── dataset_manager.py                       # データセット管理
├── background_loader.py                     # 背景埋め込み管理
├── config.json                              # 設定ファイル
├── .env                                     # 環境変数（HF Token）
├── requirements.txt                         # 依存関係
├── README.md                                # このファイル
├── prep_jvs_embeddings.py                   # JVS埋め込み作成
├── prep_common_voice_embeddings.py          # Common Voice埋め込み作成
├── create_embeddings.py                     # 埋め込み作成統合
├── background_jvs_ecapa.npz                 # JVS背景埋め込み
├── background_common_voice_ja_ecapa.npz     # Common Voice背景埋め込み
├── enrolled_speakers_embeddings.npz         # 登録話者キャッシュ
├── enroll/                                  # ユーザー登録話者
│   ├── speaker1/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── speaker2/
└── docs/                                    # ドキュメント
    ├── QUICK_START_GUIDE.md
    ├── SPEAKER_MANAGEMENT_GUIDE.md
    └── FAQ.md
```

## 🚀 クイックスタート

### 1. 依存関係のインストール

#### GPU版PyTorch（CUDA 12.6）のインストール

```bash
# 1. PyTorch（GPU版）をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 残りのライブラリをインストール  
pip install -r requirements.txt
```

### 2. Hugging Face認証設定（複数話者分析用）

```bash
# .envファイルにHugging Face Tokenを設定
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

**Hugging Face Token取得方法**:
1. https://huggingface.co/settings/tokens でトークン作成
2. https://huggingface.co/pyannote/speaker-diarization で利用規約同意

### 3. 話者登録用データの準備

```
enroll/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.wav
├── speaker2/
│   ├── audio1.wav
│   └── audio2.wav
└── speaker3/
    └── audio1.wav
```

### 4. 背景埋め込みの作成

#### 方法A: ダミーファイルで動作確認（簡単）

```bash
python create_dummy_cv_embeddings.py
```

#### 方法B: 実際のデータセットから作成（推奨）

詳細は[背景埋め込み作成ガイド](#-背景埋め込み作成ガイド)を参照

### 5. アプリケーション起動

```bash
# 仮想環境の起動（Windows）
.venv\\Scripts\\activate

# アプリ起動
streamlit run app.py
```

ブラウザで http://localhost:8501 にアクセス

## 📖 使用方法

### 🎤 単一話者識別

1. **システム初期化**
   - サイドバーの「🚀 モデル初期化」をクリック
   - モデル読み込みと話者データベース構築を待機

2. **話者識別**
   - 「🎤 単一話者識別」タブを選択
   - 音声ファイルをアップロード（WAV, MP3, FLAC, M4A, OGG対応）
   - 「🔍 話者識別開始」をクリック
   - 結果とトップ10スコア詳細を確認

### 🎭 複数話者分析

1. **システム初期化**
   - 「🎭 複数話者分析」タブを選択
   - 「🚀 複数話者分析システムを初期化」をクリック

2. **分析設定**
   - 最小・最大話者数を設定
   - JVS/Common Voice話者の表示設定を調整

3. **分析実行**
   - 複数話者音声をアップロード
   - 「🎭 複数話者分析開始」をクリック

4. **結果確認**
   - **📊 視覚化**: タイムラインチャートで話者切り替えを確認
   - **📋 時系列セグメント**: 各セグメントの詳細とトップ5スコア
   - **👥 話者別統計**: 発話時間・セグメント数・信頼度の分析

### 📊 可視化機能

#### ⏰ ダイアライゼーションタブ
- pyannote.audioの話者分離結果をタイムライン表示
- SPEAKER_00, SPEAKER_01形式のラベル
- ホバーで詳細情報（時間・認識話者・信頼度）

#### 👥 話者別タイムラインタブ  
- 認識話者ベースのタイムライン表示
- 実際の話者名（田中太郎、佐藤花子など）
- 話者別統計テーブル（発話時間・セグメント数・平均信頼度）

## 🎯 背景埋め込み作成ガイド

### 📋 JVS埋め込み作成

#### ステップ1: JVSデータセットのダウンロード

1. **JVS公式サイト**にアクセス: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
2. **Google Drive**リンクから`jvs_ver1.zip` (約2.7GB) をダウンロード
3. 適当な場所に解凍（例：`C:\\Downloads\\jvs_ver1\\`）

#### ステップ2: 埋め込み作成

```bash
# JVSのパスを指定して実行
python prep_jvs_embeddings.py C:\\Downloads\\jvs_ver1

# オプション: 話者あたりのファイル数を調整
python prep_jvs_embeddings.py C:\\Downloads\\jvs_ver1 --per-speaker 30
```

**処理時間**: CPU 30分〜1時間、GPU 10分〜20分  
**作成ファイル**: `background_jvs_ecapa.npz` (約3.8MB)

### 🌐 Common Voice埋め込み作成

```bash
# 最新版を使用（Hugging Face認証が必要）
python prep_common_voice_embeddings.py --max-samples 5000

# 認証なしで古いバージョンを使用
python prep_common_voice_embeddings.py
```

**処理時間**: 15分〜30分  
**作成ファイル**: `background_common_voice_ja_ecapa.npz` (約19MB)

### 🔧 統合スクリプト

```bash
# JVSとCommon Voice両方
python create_embeddings.py --jvs-path C:\\Downloads\\jvs_ver1 --cv-max-samples 5000

# JVSのみ
python create_embeddings.py --jvs-path C:\\Downloads\\jvs_ver1 --skip-cv
```

## 🔧 設定ファイル (config.json)

### 基本設定

```json
{
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

### 表示設定の調整

```json
{
  "ui": {
    "show_jvs_in_results": true,        // JVS話者を結果に表示
    "show_common_voice_in_results": true // Common Voice話者を結果に表示
  }
}
```

## 🔍 トラブルシューティング

### よくあるエラーと解決法

#### 1. pyannote.audio認証エラー
```
エラー: Hugging Face tokenが設定されていません
解決法:
1. https://huggingface.co/settings/tokens でトークン作成
2. https://huggingface.co/pyannote/speaker-diarization で利用規約同意
3. .envファイルにHF_TOKEN=your_token_hereを設定
```

#### 2. モジュール不足エラー
```
エラー: pyannote.audioがインストールされていません
解決法: pip install pyannote.audio>=3.1.0
```

#### 3. メモリ不足エラー
```
エラー: CUDA out of memory
解決法: 
- config.jsonのbackground_speakers_countを50に削減
- max_speakersを5に削減
```

#### 4. セグメント認識エラー
```
エラー: 'JapaneseSpeakerRecognizer' object has no attribute '_compute_similarity'
解決法: enhanced_speaker_recognition.pyが最新版か確認
```

### パフォーマンス最適化

#### GPU使用の確認
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

#### メモリ使用量の調整
- `background_speakers_count`: 100 → 50（メモリ節約）
- `max_speakers`: 10 → 5（複数話者分析）
- `min_segment_duration`: 0.5 → 1.0（短いセグメント除外）

## 🎯 重要な機能説明

### 複数話者分析の処理フロー

1. **ダイアライゼーション**: pyannote.audioで「いつ誰が話しているか」を検出
2. **セグメント切り出し**: 各時間帯の音声を16kHzに前処理
3. **話者認識**: 各セグメントを既存の登録話者から識別
4. **結果統合**: タイムライン・統計・トップ5スコア表示

### フィルタリング機能

以下の話者IDは表示設定で制御できます：
- **JVS話者**: `jvs001` ~ `jvs100`
- **Common Voice話者**: `cv_*`, `commonvoice_*` で始まるID

### AS-Norm（Adaptive Score Normalization）

```
normalized_score = (raw_score - background_mean) / background_std
```

背景データセットとの類似度を使用してZ-score正規化を行い、より信頼性の高い識別スコアを提供します。

## 📊 ファイルサイズと処理時間の目安

### システム要件
- **Python**: 3.8以上
- **メモリ**: 8GB以上推奨（複数話者分析使用時）
- **ストレージ**: 100MB（埋め込みファイル用）
- **GPU**: オプション（CUDA/MPS対応、処理速度3-5倍向上）

### 処理時間（GPU使用時）
- **単一話者識別**: 1-3秒
- **複数話者分析**: 10-30秒（音声長・話者数による）
- **ダイアライゼーション**: 5-15秒
- **話者認識**: 5-15秒（セグメント数による）

## 📝 新機能（複数話者分析）

### 主な特徴
- **pyannote.audio 3.1**による最新の話者ダイアライゼーション
- **2段階処理**：分離→認識で高精度分析
- **リアルタイム可視化**：Plotlyによるインタラクティブチャート
- **詳細分析**：各セグメントのトップ5候補表示

### 対応音声形式
- **入力**: WAV, MP3, FLAC, M4A, OGG
- **推奨**: 16kHz以上、2-30秒、複数話者が明確に分離されている音声

### 制限事項
- **最大話者数**: 10名（設定で調整可能）
- **最小セグメント長**: 0.5秒（設定で調整可能）
- **重複発話**: 部分的対応（pyannote.audioの性能に依存）

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストや問題報告を歓迎します。

## 📚 参考文献

- [SpeechBrain](https://speechbrain.github.io/)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [JVS Dataset](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [Common Voice](https://commonvoice.mozilla.org/)

## 📞 サポート

問題や質問がある場合は、GitHubのIssuesでお知らせください。