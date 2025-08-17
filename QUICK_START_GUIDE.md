# 🚀 クイックスタートガイド - 5分で始める話者認識

## 🎯 このガイドの目的

**5分で話者認識システムを動かす**ための最短手順です。詳細は他のガイドで説明します。

## ⏱️ 所要時間
- **準備**: 2分
- **動作確認**: 3分
- **合計**: 5分

## 📋 事前に必要なもの

- ✅ Python 3.8以上
- ✅ インターネット接続
- ✅ 識別したい人の音声ファイル（2〜3個）

## 🏃‍♂️ 5分クイックスタート

### Step 1: ライブラリインストール（1分）

```bash
# プロジェクトフォルダに移動
cd C:\Users\uenot\japanese-speaker-recognition

#### GPU版PyTorch（CUDA 12.6）のインストール

```bash
# 1. PyTorch（GPU版）をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. 残りのライブラリをインストール  
pip install -r requirements.txt
```


### Step 2: 話者データ準備（1分）

#### フォルダ作成
```bash
mkdir enroll\テスト話者
```

#### 音声ファイル配置
あなたの音声ファイル（WAV, MP3など）を `enroll\テスト話者\` にコピー

```
enroll\
└── テスト話者\
    ├── 音声1.wav
    ├── 音声2.wav
    └── 音声3.wav
```

### Step 4: アプリ起動（30秒）

```bash
 streamlit run app.py          
```

ブラウザが自動で開く（開かない場合は http://localhost:8501 にアクセス）

### Step 5: 初期化（1分）

1. **サイドバー**の「🚀 モデル初期化」をクリック
2. **1分程度待つ**（初回のみモデルダウンロード）
3. 「モデル初期化完了」と表示されれば成功

### Step 6: 話者識別テスト（1分）

1. **「🎤 話者識別」タブ**をクリック
2. **音声ファイルをアップロード**（ドラッグ&ドロップ）
3. **「🔍 話者識別開始」**をクリック
4. **結果確認**

## 🎉 成功例

### 結果画面
```
🎯 識別結果

👤 話者: テスト話者
📊 信頼度: 0.892
📈 生スコア: 0.847

📋 全話者スコア詳細:
1. テスト話者    ████████████ 0.892
```

## 🔧 うまくいかない場合

### エラー1: モジュールが見つからない
```bash
# 解決法
pip install -r requirements.txt
```

### エラー2: 話者が表示されない
```bash
# 音声ファイルの場所確認
dir enroll\テスト話者
```

### エラー3: 識別結果が出ない
- 音声ファイルが2秒以上30秒以下か確認
- WAV, MP3, FLAC, M4A, OGG形式か確認

## 📚 次のステップ

### 🎯 より高精度にしたい場合

1. **JVS埋め込み作成** → [背景埋め込み作成ガイド（README.md）](README.md#-背景埋め込み作成ガイド)
2. **Common Voice埋め込み作成** → [README.md](README.md#-common-voice埋め込み作成)

### 👥 複数話者を管理したい場合

**話者管理ガイド** → [SPEAKER_MANAGEMENT_GUIDE.md](SPEAKER_MANAGEMENT_GUIDE.md)

### ⚙️ 設定をカスタマイズしたい場合

**README.md** → [設定ファイル説明](README.md#-設定ファイル-configjson)

## 🎯 このクイックスタートで体験できること

- ✅ 話者識別の基本機能
- ✅ Webアプリの操作感
- ✅ 識別精度の目安
- ✅ システムの動作確認

## 💡 重要な注意点

### ダミーデータについて
```
⚠️ 今回使用した background_common_voice_ja_ecapa.npz は
   ダミーデータです。

✅ 実際のプロジェクトでは本物のデータを使用することを
   強く推奨します。
```

### 本格運用への移行
```bash
# ダミーファイルを削除
del background_common_voice_ja_ecapa.npz

# 本物のCommon Voice埋め込み作成
python prep_common_voice_embeddings.py

# JVS埋め込み作成（オプション）
python prep_jvs_embeddings.py C:\path\to\jvs
```

## 🎊 おめでとうございます！

**5分で話者認識システムが動きました！**

これで基本的な動作が確認できたので、次は：
- より多くの話者を追加
- 高精度な背景モデルの作成
- 実際の業務での活用

を検討してみてください。

---

### 📞 困ったときは

1. **README.md** - 詳細な機能説明
2. **SPEAKER_MANAGEMENT_GUIDE.md** - 話者管理の詳細
3. **GitHub Issues** - 問題報告