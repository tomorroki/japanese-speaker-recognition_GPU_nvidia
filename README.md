# 日本語話者認識システム

SpeechBrain + ECAPA-TDNNとpyannote.audioを使用した高精度な日本語話者識別・分析Webアプリケーション  
**🎵 リアルタイム音声同期対応 | 🔧 手動話者分離機能 | 📊 高度可視化**

## 🎯 概要

このシステムは、**SpeechBrain**の**ECAPA-TDNN**モデルと**pyannote.audio**を組み合わせて、音声ファイルから話者を高精度で識別・分析するWebアプリケーションです。単一話者の識別、複数話者の自動分析、そして手動による精密な話者分離機能を提供します。

## ✨ 主な機能

### 🎤 単一話者識別
- **高精度認識**: ECAPA-TDNNによる最先端の話者埋め込み技術
- **AS-Norm対応**: 背景モデルを使用したスコア正規化で精度向上
- **トップ10スコア表示**: 全候補話者のスコア詳細分析

### 🎭 複数話者分析
- **話者ダイアライゼーション**: pyannote.audioによる「いつ誰が話しているか」検出
- **時系列話者認識**: 各時間帯の話者を自動識別
- **インタラクティブ可視化**: Ganttチャートとタイムライングラフ
- **トップ5表示**: 各セグメントの認識候補をランキング表示

### 🔧 手動話者分離（NEW!）
- **🎵 リアルタイム音声同期**: streamlit-advanced-audio (audix) による真の位置同期
- **📍 範囲選択**: ドラッグ選択と数値入力による精密な時間指定
- **⚡ 瞬時セグメント作成**: 選択範囲から即座にセグメント生成
- **🎯 手動話者割当**: セグメント単位での詳細な話者管理
- **📊 統計分析**: 話者別発話時間とセグメント統計

### 📊 可視化・分析機能
- **インタラクティブ波形表示**: WaveSurfer.js基盤の高度音声プレーヤー
- **リアルタイム同期**: 再生位置とプログレスバーの完全同期
- **話者別Ganttチャート**: 時系列での話者分布可視化
- **統計ダッシュボード**: セグメント数・発話時間・信頼度の詳細分析
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
- **streamlit-advanced-audio** (高度音声プレーヤー)
- **PyTorch** (深層学習フレームワーク)
- **librosa, soundfile** (音声処理)
- **scikit-learn** (機械学習)
- **Plotly** (可視化)
- **Hugging Face** (モデル・データセット)

## 📁 プロジェクト構造

```
japanese-speaker-recognition_GPU_nvidia/
├── app.py                                    # 4タブStreamlit メインアプリ
├── enhanced_speaker_recognition.py           # 単一話者認識システム
├── speaker_diarization.py                   # 複数話者分析システム
├── manual_speaker_segmentation_v2.py        # 手動話者分離システム（v2）
├── segmentation_core.py                     # セグメンテーション基盤
├── segmentation_processors.py               # 専門処理クラス群
├── segment_processor.py                     # セグメント処理
├── dataset_manager.py                       # データセット管理
├── background_loader.py                     # 背景埋め込み管理
├── error_handling.py                        # エラーハンドリング統合
├── logging_config.py                        # ログシステム
├── performance_optimizer.py                 # パフォーマンス最適化
├── type_definitions.py                      # 型定義集約
├── config.json                              # 統合設定ファイル
├── .env                                     # 環境変数（HF Token）
├── requirements.txt                         # 依存関係（audix含む）
├── README.md                                # このファイル
├── FILE_DOCUMENTATION.md                    # ファイル構成詳細
├── CLAUDE.md                                # Claude Code開発ガイド
├── prep_jvs_embeddings.py                   # JVS埋め込み作成
├── prep_common_voice_embeddings.py          # Common Voice埋め込み作成
├── create_embeddings.py                     # 埋め込み作成統合
├── background_jvs_ecapa.npz                 # JVS背景埋め込み（3.8MB）
├── background_common_voice_ja_ecapa.npz     # Common Voice背景埋め込み（19MB）
├── enrolled_speakers_embeddings.npz         # 登録話者キャッシュ
├── enroll/                                  # ユーザー登録話者
│   ├── speaker1/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── speaker2/
├── logs/                                    # ログファイル
├── error_reports/                           # エラーレポート
└── background_embeddings/                   # 追加背景データ
```

## 🚀 クイックスタート

### 1. 依存関係のインストール

#### GPU版PyTorch（CUDA 12.6）のインストール

```bash
# 1. PyTorch（GPU版）をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 残りのライブラリをインストール（audix含む）
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
   - **📊 視覚化**: Ganttチャートで話者切り替えを確認
   - **📋 時系列セグメント**: 各セグメントの詳細とトップ5スコア
   - **👥 話者別統計**: 発話時間・セグメント数・信頼度の分析

### 🔧 手動話者分離

1. **音声読み込み**
   - 「🔧 手動話者分離」タブを選択
   - 音声ファイルをアップロード

2. **範囲選択（2つの方法）**
   - **方法A**: 波形上でドラッグして範囲選択
   - **方法B**: 数値入力で精密な時間指定

3. **セグメント作成**
   - 「➕ セグメント追加」でセグメント生成
   - リアルタイム音声同期で再生位置を確認

4. **話者割当と分析**
   - 各セグメントに話者を手動割当
   - 認識結果で候補確認
   - 統計情報でデータ分析

### 📊 可視化機能

#### 🎵 リアルタイム音声プレーヤー
- **streamlit-advanced-audio (audix)** による高度プレーヤー
- **真のリアルタイム同期**: currentTime と isPlaying の完全連携
- **WaveSurfer.js**: プロ級の波形表示とインタラクション
- **範囲選択**: ドラッグによる直感的操作
- **セグメント可視化**: 既存セグメントのハイライト表示

#### 📊 Ganttスタイルチャート
- **話者別タイムライン**: 時系列での発話パターン可視化
- **セグメント統計**: 発話時間・セグメント数・信頼度分析
- **インタラクティブ**: ホバーで詳細情報表示
- **フィルタリング**: 話者カテゴリ別表示制御

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

#### 4. audix関連エラー
```
エラー: streamlit-advanced-audioがインストールされていません
解決法: pip install streamlit-advanced-audio>=0.1.0
```

#### 5. セグメント認識エラー
```
エラー: 'AudioSegmentProcessor' object has no attribute 'extract_segment'
解決法: segmentation_processors.pyが最新版か確認
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
- `max_parallel_workers`: 4 → 2（並列処理制限）

#### audix最適化
- `wavesurfer_options.height`: 200px（標準）
- `bar_width`: 1（詳細表示）
- `normalize`: true（波形正規化）

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
- **手動話者分離**: リアルタイム（即座の範囲選択と再生）
- **ダイアライゼーション**: 5-15秒
- **話者認識**: 5-15秒（セグメント数による）
- **audix音声同期**: <100ms（ほぼ遅延なし）

## 🚀 最新機能ハイライト

### 🎵 真のリアルタイム音声同期（NEW!）
- **streamlit-advanced-audio (audix)** による完全な位置同期
- **WaveSurfer.js基盤**: プロレベルの波形表示とインタラクション
- **position tracking**: `currentTime` と `isPlaying` のライブ同期
- **範囲選択**: ドラッグ&ドロップによる直感的な時間範囲指定

### 🔧 手動話者分離システム（NEW!）
- **責任分離アーキテクチャ**: 専門クラスによる模块化設計
- **リアルタイムフィードバック**: 即座の範囲選択と音声再生
- **精密な時間制御**: 0.1秒単位での正確なセグメント作成
- **統合ワークフロー**: 選択→作成→認識→割当の seamless 連携

### 📊 高度な可視化（強化済み）
- **Ganttチャート**: 話者別タイムライン表示
- **統計ダッシュボード**: 発話時間・セグメント数・信頼度分析
- **インタラクティブチャート**: Plotlyによる詳細分析
- **リアルタイム更新**: UI操作に即座に反応する動的表示

### 対応音声形式
- **入力**: WAV, MP3, FLAC, M4A, OGG
- **単一話者**: 2-30秒、明瞭な発話
- **複数話者**: 制限なし、2-10話者対応
- **手動分離**: 制限なし、任意長での精密分析

### アーキテクチャの特徴
- **型安全設計**: Protocol、TypeVar活用
- **エラーハンドリング**: 統一的な例外処理とリトライ機能
- **並列処理**: 非同期対応による高速処理
- **キャッシュ最適化**: メモリ効率とパフォーマンス向上

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