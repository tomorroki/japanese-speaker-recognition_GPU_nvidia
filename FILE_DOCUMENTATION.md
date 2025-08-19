# 📁 日本語話者認識システム - ファイル構成ドキュメント

このドキュメントでは、プロジェクト内の各ファイルとディレクトリの役割を詳しく説明します。

## 🎯 メインアプリケーション

### `app.py`
**役割**: Streamlit Webアプリケーションのメインファイル  
**機能**:
- 4つのタブ構成：🎤単一話者認識、🎭複数話者分析、🔧手動話者分離、👥話者管理
- streamlit-advanced-audio (audix) による高度音声プレーヤー統合
- リアルタイム音声位置同期と範囲選択機能
- セッション状態管理とキャッシュ制御
- モデル初期化とGPU検出

**重要な実装**:
- `display_interactive_waveform()`: audix音声プレーヤーとWaveSurfer.js統合
- `sync_audio_player_state()`: リアルタイム位置同期
- `handle_region_selection()`: 範囲選択処理

---

## 🧠 コア認識エンジン

### `enhanced_speaker_recognition.py`
**役割**: メイン話者認識エンジン  
**機能**:
- SpeechBrain ECAPA-TDNN モデル統合
- AS-Norm (Adaptive Score Normalization) 実装
- 話者エンベディング生成と類似度計算
- 背景モデルを用いた正規化処理
- キャッシュ機能付き話者データベース管理

**主要クラス**: `JapaneseSpeakerRecognizer`

### `speaker_diarization.py`
**役割**: 複数話者分析システム  
**機能**:
- pyannote.audio による話者分離
- 2段階処理：分離→認識
- タイムライン分析とセグメント統合
- Ganttチャート形式での結果表示

**主要クラス**: `SpeakerDiarizer`, `MultiSpeakerRecognizer`

---

## 🔧 手動分離システム（リファクタリング版）

### `manual_speaker_segmentation_v2.py`
**役割**: 手動話者分離の新世代実装  
**機能**:
- 責任分離アーキテクチャによる設計
- 各専門クラスへの処理委譲
- 強化されたエラーハンドリング
- 非同期処理対応
- 統計情報とエクスポート機能

**設計思想**: 単一責任原則に基づく模块化設計

### `manual_speaker_segmentation.py`
**役割**: 旧版実装（互換性維持）  
**状態**: 非推奨、v2で機能強化済み

---

## 🏗️ アーキテクチャコア

### `segmentation_core.py`
**役割**: セグメンテーション機能の基盤定義  
**機能**:
- データクラス定義：`AudioSegment`, `OperationResult`
- プロトコル定義：`AudioProcessor`, `Recognizer`
- バリデーション機能：`SegmentValidator`
- 専門クラス：`AudioLoader`, `SegmentManager`

**設計**: 型安全性とProtocolによるインターフェース設計

### `segmentation_processors.py`
**役割**: セグメンテーション処理の専門実装  
**機能**:
- `LibrosaAudioProcessor`: librosa音声処理実装
- `AudioSegmentProcessor`: キャッシュ機能付きセグメント処理
- `RecognitionManager`: 認識処理統合（同期・非同期）
- `ExportManager`: CSV/JSON/SRT形式エクスポート

### `segment_processor.py`
**役割**: 旧版セグメント処理  
**状態**: `segmentation_processors.py`で統合済み

---

## 🎛️ 設定・管理システム

### `config.json`
**役割**: 統合設定ファイル  
**設定項目**:
- モデル設定（ECAPA-TDNN, pyannote）
- 音声パラメータ（サンプルレート、長さ制限）
- 認識設定（閾値、正規化）
- UI設定（表示オプション）

### `config_unified.json`
**役割**: 旧統合設定ファイル  
**状態**: `config.json`に統合済み

### `config_manager.py`
**役割**: 設定管理ユーティリティ  
**状態**: 現在未使用、将来の機能拡張用

---

## 🗃️ データ管理

### `dataset_manager.py`
**役割**: データセット・話者管理  
**機能**:
- JVS/Common Voiceスピーカー分類
- 音声ファイル発見と検証
- 除外ルール管理
- 話者カテゴリ判定

### `background_loader.py`
**役割**: 背景エンベディング管理  
**機能**:
- 事前計算済みエンベディング読み込み
- JVS/Common Voice背景データ統合
- AS-Norm用背景モデル提供
- フォールバック処理

---

## 📊 データ準備スクリプト

### `prep_jvs_embeddings.py`
**役割**: JVSデータセット用エンベディング作成  
**機能**: JVS話者のECAPA-TDNNエンベディング事前計算

### `prep_common_voice_embeddings.py` 
**役割**: Common Voiceデータセット用エンベディング作成  
**機能**: Common Voice話者のエンベディング作成（Hugging Face認証要）

### `create_embeddings.py`
**役割**: 統合エンベディング作成スクリプト  
**機能**: JVS/Common Voice両方のエンベディング一括作成

### `create_dummy_cv_embeddings.py`
**役割**: ダミーエンベディング作成  
**機能**: テスト用の軽量エンベディング生成

---

## 🛠️ ユーティリティ・基盤

### `type_definitions.py`
**役割**: 型定義集約  
**機能**:
- TypeVar, Protocol定義
- 型安全性向上
- ジェネリック型サポート
- バリデーション機能

### `error_handling.py`
**役割**: エラーハンドリング統合システム  
**機能**:
- カスタム例外クラス
- エラーカテゴリ分類
- リトライ機能
- デコレータベースエラー処理

### `logging_config.py`
**役割**: ログシステム設定  
**機能**:
- 構造化ログ出力
- ファイル別ログ分離
- ログレベル管理
- JSON形式ログサポート

### `logging_config.json`
**役割**: ログ設定ファイル  
**設定**: ログレベル、出力先、フォーマット

### `performance_optimizer.py`
**役割**: パフォーマンス最適化  
**機能**:
- メモリ管理
- プロファイリング
- キャッシュ最適化
- 並列処理制御

---

## 📄 データファイル

### `enrolled_speakers_embeddings.npz`
**役割**: 話者データベースキャッシュ  
**内容**: `enroll/`フォルダ内話者の事前計算済みエンベディング

### `background_jvs_ecapa.npz`
**役割**: JVS背景エンベディング（3.8MB）  
**用途**: AS-Norm正規化用背景モデル

### `background_common_voice_ja_ecapa.npz` 
**役割**: Common Voice背景エンベディング（19MB）  
**用途**: AS-Norm正規化用背景モデル

### `requirements.txt`
**役割**: Python依存関係定義  
**重要**: streamlit-advanced-audio>=0.1.0 含む

---

## 📁 ディレクトリ構造

### `enroll/`
**役割**: 話者登録ディレクトリ  
**構造**:
```
enroll/
├── speaker_name/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── sample3.wav
```
**要件**: 話者あたり2-5サンプル推奨

### `logs/`
**役割**: ログファイル格納  
**ファイル**:
- `application.log`: アプリケーション全般
- `errors.log`: エラー専用
- `segmentation.log`: セグメンテーション処理

### `error_reports/`
**役割**: エラーレポート格納  
**状態**: 現在空（将来の機能拡張用）

### `background_embeddings/`
**役割**: 追加背景エンベディング格納  
**状態**: 現在空（将来の機能拡張用）

---

## 📚 ドキュメンテーション

### `README.md`
**役割**: プロジェクト概要とセットアップガイド

### `CLAUDE.md`
**役割**: Claude Code向け開発ガイド  
**内容**: アーキテクチャ、コマンド、設計思想

### `CLAUDE_JP.md`
**役割**: 日本語版開発ガイド

### `QUICK_START_GUIDE.md`
**役割**: クイックスタートガイド

### `SPEAKER_MANAGEMENT_GUIDE.md`
**役割**: 話者管理詳細ガイド

### `FAQ.md`
**役割**: よくある質問と回答

---

## 🗑️ 削除推奨ファイル

### `=0.1.0`
**役割**: 不明（誤作成ファイル）  
**推奨**: 削除

### `temp_audio_*.wav` (21個)
**役割**: 一時音声ファイル  
**推奨**: 削除（自動生成される）

---

## 🎯 システム全体の設計思想

1. **責任分離**: 各ファイルが明確な役割を持つ
2. **型安全性**: Protocol、TypeVar活用
3. **エラーハンドリング**: 統一的なエラー処理
4. **パフォーマンス**: キャッシュとGPU最適化
5. **拡張性**: プラグアブルなアーキテクチャ
6. **リアルタイム同期**: audixによる真の音声同期

このアーキテクチャにより、保守性、拡張性、パフォーマンスを両立した日本語話者認識システムを実現しています。