# 日本語話者認識システム

SpeechBrain + ECAPA-TDNNを使用した高精度な日本語話者識別Webアプリケーション

## 🎯 概要

このシステムは、**SpeechBrain**の**ECAPA-TDNN**モデルを使用して、音声ファイルから話者を高精度で識別するWebアプリケーションです。JVSとCommon Voice日本語データセットの話者を背景モデルとして活用し、AS-Norm（Adaptive Score Normalization）により識別精度を向上させています。

## ✨ 主な機能

- **🎤 高精度話者識別**: ECAPA-TDNNによる最先端の話者埋め込み技術
- **🗂️ 背景話者除外**: JVS・Common Voice話者の自動除外機能
- **📊 AS-Norm対応**: 背景モデルを使用したZ-score正規化でスコア改善
- **💾 軽量配布**: 音声ファイルではなく埋め込みベクトル（.npz）のみを同梱
- **🖥️ 直感的UI**: Streamlitによる使いやすいWebインターface
- **⚙️ 柔軟な設定**: JSON設定ファイルによるパラメータ調整
- **📈 詳細分析**: 全話者スコア表示と統計情報

## 🛠️ 技術スタック

- **Python 3.8+**
- **SpeechBrain** (speechbrain/spkrec-ecapa-voxceleb)
- **Streamlit** (Web UI)
- **PyTorch** (深層学習フレームワーク)
- **librosa, soundfile** (音声処理)
- **scikit-learn** (機械学習)
- **Plotly** (可視化)
- **Hugging Face Datasets** (Common Voice ストリーミング)

## 📁 プロジェクト構造

```
japanese-speaker-recognition/
├── app.py                                    # Streamlit メインアプリ
├── enhanced_speaker_recognition.py           # 核となる認識システム
├── dataset_manager.py                        # データセット管理
├── background_loader.py                      # 背景埋め込み管理
├── config.json                               # 設定ファイル
├── requirements.txt                          # 依存関係
├── README.md                                 # このファイル
├── prep_jvs_embeddings.py                    # JVS埋め込み作成スクリプト
├── prep_common_voice_embeddings.py           # Common Voice埋め込み作成スクリプト
├── create_embeddings.py                      # 埋め込み作成統合スクリプト
├── create_dummy_cv_embeddings.py             # ダミー埋め込み作成（テスト用）
├── background_jvs_ecapa.npz                  # JVS背景埋め込み（作成後）
├── background_common_voice_ja_ecapa.npz      # Common Voice背景埋め込み（作成後）
├── enroll/                                   # ユーザー登録話者（識別対象）
│   ├── yamada_taro/                          # 話者1のディレクトリ
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── sato_hanako/                          # 話者2のディレクトリ
│   └── tanaka_jiro/                          # 話者3のディレクトリ
├── background_datasets/                      # 背景モデル用（識別対象外）
│   ├── jvs/                                 # JVSデータセット（音声ファイル）
│   └── common_voice_ja/                     # Common Voiceデータセット（音声ファイル）
└── background_embeddings/                    # 埋め込みファイル配置用（オプション）
    ├── background_jvs_ecapa.npz
    └── background_common_voice_ja_ecapa.npz
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


### 2. 話者登録用データの準備

識別したい話者の音声ファイルを`enroll/`フォルダに配置：

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

### 3. 背景埋め込みの作成

#### 方法A: ダミーファイルで動作確認（簡単）

```bash
python create_dummy_cv_embeddings.py
```

#### 方法B: 実際のデータセットから作成（推奨）

詳細は[背景埋め込み作成ガイド](#-背景埋め込み作成ガイド)を参照

### 4. アプリケーション起動

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 にアクセス

## 🎯 背景埋め込み作成ガイド

背景埋め込みを作成することで、AS-Norm（Adaptive Score Normalization）による高精度な話者識別が可能になります。

### 📋 JVS埋め込み作成

#### ステップ1: JVSデータセットのダウンロード

1. **JVS公式サイト**にアクセス: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
2. **Google Drive**リンクから`jvs_ver1.zip` (約2.7GB) をダウンロード
3. 適当な場所に解凍（例：`C:\Downloads\jvs_ver1\`）

#### ステップ2: 埋め込み作成

```bash
# JVSのパスを指定して実行（パスは実際の解凍先に変更）
python prep_jvs_embeddings.py C:\Downloads\jvs_ver1

# オプション: 話者あたりのファイル数を調整
python prep_jvs_embeddings.py C:\Downloads\jvs_ver1 --per-speaker 30
```

**処理時間**: CPU 30分〜1時間、GPU 10分〜20分  
**作成ファイル**: `background_jvs_ecapa.npz` (約3.8MB)

#### ステップ3: 音声ファイルの削除（任意）

埋め込み作成後、元の音声ファイル（2.7GB）は削除しても構いません。埋め込みファイル（3.8MB）のみでアプリは動作します。

### 🌐 Common Voice埋め込み作成

#### 方法1: 認証なしで古いバージョンを使用

```bash
python prep_common_voice_embeddings.py
```

自動的に利用可能なバージョンにフォールバックします。

#### 方法2: 最新版を使用（Hugging Face認証が必要）

1. **Hugging Faceアカウント**を作成: https://huggingface.co/
2. **Common Voiceデータセット**へのアクセス申請: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
3. **認証設定**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
4. **埋め込み作成**:
   ```bash
   python prep_common_voice_embeddings.py --max-samples 5000
   ```

**処理時間**: 15分〜30分  
**作成ファイル**: `background_common_voice_ja_ecapa.npz` (約19MB)

#### 方法3: ダミーファイルで動作確認

```bash
python create_dummy_cv_embeddings.py
```

### 🔧 統合スクリプト

両方の埋め込みを一度に作成：

```bash
# JVSとCommon Voice両方
python create_embeddings.py --jvs-path C:\Downloads\jvs_ver1 --cv-max-samples 5000

# JVSのみ
python create_embeddings.py --jvs-path C:\Downloads\jvs_ver1 --skip-cv

# Common Voiceのみ
python create_embeddings.py --skip-jvs --cv-max-samples 3000
```

## 🔧 設定ファイル (config.json)

### 基本設定

```json
{
  \"recognition\": {
    \"threshold\": 0.25,                    // 識別しきい値
    \"use_score_normalization\": true,      // AS-Norm の有効/無効
    \"background_speakers_count\": 100      // 背景話者サンプル数
  },
  \"audio\": {
    \"sample_rate\": 16000,                 // サンプリングレート
    \"min_duration\": 2.0,                  // 最小音声長（秒）
    \"max_duration\": 30.0,                 // 最大音声長（秒）
    \"normalize\": true                     // 音声正規化
  },
  \"datasets\": {
    \"exclude_background_speakers\": true,  // 背景話者除外機能
    \"allow_jvs_speakers\": false,          // JVS話者を識別候補に含める
    \"allow_common_voice_speakers\": false  // Common Voice話者を識別候補に含める
  }
}
```

### JVS話者を識別候補に含める方法

JVS話者（jvs001〜jvs100）を識別対象にしたい場合：

```json
{
  \"datasets\": {
    \"allow_jvs_speakers\": true,           // これをtrueに変更
    \"allow_common_voice_speakers\": false
  }
}
```

### 背景話者除外を完全無効化

すべての話者を識別候補にしたい場合：

```json
{
  \"datasets\": {
    \"exclude_background_speakers\": false  // これをfalseに変更
  }
}
```

## 📖 使用方法

### 1. システム初期化
1. Webアプリにアクセス
2. サイドバーの「🚀 モデル初期化」をクリック
3. モデル読み込みと話者データベース構築を待機

### 2. 話者識別
1. 「🎤 話者識別」タブを選択
2. 音声ファイルをアップロード（WAV, MP3, FLAC, M4A, OGG対応）
3. 「🔍 話者識別開始」をクリック
4. 結果とスコア詳細を確認

### 3. 話者管理
1. 「👥 話者管理」タブで登録状況を確認
2. 除外された話者と理由を表示
3. 音声ファイル数の統計を確認

### 4. システム情報
1. 「📊 統計情報」タブでシステム情報を確認
2. パフォーマンス情報を表示

## 🎯 重要な機能説明

### 背景話者除外機能

以下の話者IDは自動的に識別候補から除外されます（設定で変更可能）：

- **JVS話者**: `jvs001` ~ `jvs100`
- **Common Voice話者**: `cv_*`, `commonvoice_*` で始まるID

この機能により、公開データセットの話者と実際のユーザーを明確に分離できます。除外したくない場合は`config.json`で設定変更できます。

### AS-Norm（Adaptive Score Normalization）

背景データセットとの類似度を使用してZ-score正規化を行い、より信頼性の高い識別スコアを提供します：

```
normalized_score = (raw_score - background_mean) / background_std
```

### 軽量配布システム

- **JVS**: 音声ファイル（2.7GB）→ 埋め込み（3.8MB）に圧縮
- **Common Voice**: ストリーミング取得 → 埋め込み（19MB）で軽量化
- **再配布**: 音声ファイルではなく埋め込みのみを同梱（ライセンス対応）

### 音声品質管理

- **音声長制限**: 2秒～30秒の範囲で処理
- **フォーマット統一**: 16kHz モノラルに自動変換
- **正規化**: 音声レベルの自動調整

## 🔧 トラブルシューティング

### よくあるエラーと解決法

#### 1. Common Voice認証エラー
```
エラー: Dataset 'mozilla-foundation/common_voice_17_0' is a gated dataset
解決法:
1. https://huggingface.co/ でアカウント作成
2. データセットへのアクセス申請
3. huggingface-cli login で認証
または: python prep_common_voice_embeddings.py （古いバージョンで自動フォールバック）
```

#### 2. JVS埋め込み作成エラー
```
エラー: No such file or directory
解決法: python prep_jvs_embeddings.py [正しいJVSパス]
例: python prep_jvs_embeddings.py C:\Downloads\jvs_ver1
```

#### 3. メモリ不足エラー
```
エラー: CUDA out of memory
解決法: python prep_jvs_embeddings.py [パス] --per-speaker 25
```

#### 4. 背景埋め込みが見つからない
```
解決法: python create_dummy_cv_embeddings.py でダミーファイル作成
```

### パフォーマンス最適化

#### GPU使用の確認
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
```

#### メモリ使用量の調整
- `background_speakers_count`: 100 → 50（メモリ節約）
- `max_samples`: 5000 → 2000（Common Voice）
- `per_speaker`: 50 → 25（JVS）

## 🔍 API使用例

システムをプログラムから使用する場合：

```python
from enhanced_speaker_recognition import JapaneseSpeakerRecognizer

# 初期化
recognizer = JapaneseSpeakerRecognizer()
recognizer.initialize_model()

# 話者データベース構築
recognizer.build_speaker_database()

# 背景モデル構築
recognizer.build_background_model()

# 話者識別
result = recognizer.recognize_speaker(\"test_audio.wav\")

if result:
    print(f\"識別された話者: {result.speaker_id}\")
    print(f\"信頼度: {result.confidence:.3f}\")
    print(f\"生スコア: {result.raw_score:.3f}\")
```

## 📊 ファイルサイズと処理時間の目安

### JVS埋め込み
- **元データ**: 2.7GB（音声ファイル）
- **埋め込み**: 3.8MB（.npzファイル）
- **処理時間**: CPU 30分〜1時間、GPU 10分〜20分
- **話者数**: 100名
- **サンプル数**: 5,000発話（デフォルト）

### Common Voice埋め込み
- **元データ**: ストリーミング（全量DL不要）
- **埋め込み**: 19MB（.npzファイル）
- **処理時間**: 15分〜30分
- **サンプル数**: 5,000発話（デフォルト）

### システム要件
- **Python**: 3.8以上
- **メモリ**: 4GB以上推奨
- **ストレージ**: 100MB（埋め込みファイル用）
- **GPU**: オプション（CUDA/MPS対応）

## 📁 ファイル管理

### 作成されるファイル
```
japanese-speaker-recognition/
├── background_jvs_ecapa.npz              # JVS埋め込み（3.8MB）
├── background_common_voice_ja_ecapa.npz  # Common Voice埋め込み（19MB）
└── ...
```

### バックアップ推奨
- 埋め込みファイル（.npz）は作成に時間がかかるため、バックアップ推奨
- `config.json`の設定もバックアップ推奨

### クリーンアップ
```bash
# 元の音声ファイルを削除（埋め込み作成後）
rm -rf jvs_ver1/  # JVS音声ファイル（2.7GB）

# 一時ファイルを削除
rm -rf __pycache__/
```

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プルリクエストや問題報告を歓迎します。以下の手順でご協力ください：

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📚 参考文献

- [SpeechBrain](https://speechbrain.github.io/)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [JVS Dataset](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [Common Voice](https://commonvoice.mozilla.org/)

## 📞 サポート

問題や質問がある場合は、GitHubのIssuesでお知らせください。