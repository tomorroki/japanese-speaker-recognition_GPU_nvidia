"""
日本語話者認識システム
SpeechBrain + ECAPA-TDNNを使用した高精度話者識別
"""

import os
import json
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from speechbrain.pretrained import EncoderClassifier
from dataset_manager import DatasetManager
from background_loader import BackgroundEmbeddingLoader

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class RecognitionResult:
    """認識結果を格納するデータクラス"""
    speaker_id: str
    confidence: float
    raw_score: float
    normalized_score: Optional[float] = None
    all_scores: Optional[Dict[str, float]] = None

class JapaneseSpeakerRecognizer:
    """日本語話者認識システム"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # デバイス設定
        self.device = self._setup_device()
        
        # データセット管理
        self.dataset_manager = DatasetManager(config_path)
        
        # 背景埋め込み管理
        self.background_loader = BackgroundEmbeddingLoader(config=self.config)
        
        # モデル設定
        self.model = None
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.min_duration = self.config["audio"]["min_duration"]
        self.max_duration = self.config["audio"]["max_duration"]
        
        # 話者データベース
        self.speaker_embeddings = {}
        self.background_embeddings = []
        self.scaler = StandardScaler() if self.config["recognition"]["use_score_normalization"] else None
        
        # しきい値
        self.threshold = self.config["recognition"]["threshold"]
        
        self.logger.info("JapaneseSpeakerRecognizer initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config["logging"]["format"])
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_device(self) -> torch.device:
        """デバイスの設定"""
        device_config = self.config["model"]["device"]
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info("CUDA device detected, using GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("MPS device detected, using Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
            self.logger.info(f"Using configured device: {device}")
        
        return device
    
    def initialize_model(self) -> bool:
        """モデルの初期化"""
        try:
            self.logger.info("SpeechBrainモデルを読み込み中...")
            
            self.model = EncoderClassifier.from_hparams(
                source=self.config["model"]["name"],
                run_opts={"device": self.device}
            )
            
            self.logger.info("モデルの読み込み完了")
            return True
            
        except Exception as e:
            self.logger.error(f"モデル初期化に失敗: {e}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        音声ファイルの前処理
        
        Args:
            audio_path: 音声ファイルのパス
            
        Returns:
            前処理済み音声テンソル、失敗時はNone
        """
        try:
            # 音声ファイル読み込み
            waveform, sr = torchaudio.load(audio_path)
            
            # ステレオからモノラルに変換
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # リサンプリング
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 長さチェック
            duration = waveform.shape[1] / self.sample_rate
            
            if duration < self.min_duration:
                self.logger.warning(f"音声が短すぎます: {duration:.2f}秒 (最小: {self.min_duration}秒)")
                return None
            
            if duration > self.max_duration:
                # 最大長でクリップ
                max_samples = int(self.max_duration * self.sample_rate)
                waveform = waveform[:, :max_samples]
                self.logger.info(f"音声を{self.max_duration}秒でクリップしました")
            
            # 正規化
            if self.config["audio"]["normalize"]:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform.squeeze(0)  # (1, length) -> (length,)
            
        except Exception as e:
            self.logger.error(f"音声前処理に失敗: {audio_path}, エラー: {e}")
            return None
    
    def extract_embedding(self, audio_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        音声から話者埋め込みを抽出
        
        Args:
            audio_tensor: 音声テンソル
            
        Returns:
            話者埋め込みベクトル、失敗時はNone
        """
        try:
            if self.model is None:
                self.logger.error("モデルが初期化されていません")
                return None
            
            # バッチ次元を追加
            audio_batch = audio_tensor.unsqueeze(0).to(self.device)
            
            # 埋め込み抽出
            with torch.no_grad():
                embeddings = self.model.encode_batch(audio_batch)
                embedding = embeddings.squeeze(0).cpu().numpy()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"埋め込み抽出に失敗: {e}")
            return None
    
    def enroll_speaker_from_files(self, speaker_id: str, audio_files: List[str]) -> bool:
        """
        複数音声ファイルから話者を登録
        
        Args:
            speaker_id: 話者ID
            audio_files: 音声ファイルのリスト
            
        Returns:
            成功時True
        """
        try:
            embeddings = []
            
            for audio_file in audio_files:
                audio_tensor = self.preprocess_audio(audio_file)
                if audio_tensor is None:
                    continue
                
                embedding = self.extract_embedding(audio_tensor)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                self.logger.error(f"話者 {speaker_id} の有効な音声ファイルが見つかりません")
                return False
            
            # 複数埋め込みの平均を取る
            mean_embedding = np.mean(embeddings, axis=0)
            self.speaker_embeddings[speaker_id] = mean_embedding
            
            self.logger.info(f"話者 {speaker_id} を登録: {len(embeddings)}ファイルから埋め込み生成")
            return True
            
        except Exception as e:
            self.logger.error(f"話者登録に失敗: {speaker_id}, エラー: {e}")
            return False
    
    def build_speaker_database(self, enroll_dir: str = "enroll", use_cache: bool = True) -> int:
        """
        話者データベースの構築
        
        Args:
            enroll_dir: 登録用ディレクトリ
            use_cache: キャッシュを使用するかどうか
            
        Returns:
            登録された話者数
        """
        if self.model is None:
            self.logger.error("モデルが初期化されていません")
            return 0
        
        cache_file = self.config.get("cache", {}).get("cache_file", "enrolled_speakers_embeddings.npz")
        use_cache_config = self.config.get("cache", {}).get("use_speaker_embedding_cache", True)
        auto_save = self.config.get("cache", {}).get("auto_save_cache", True)
        
        # 設定とパラメータの両方でキャッシュ使用を判定
        use_cache = use_cache and use_cache_config
        
        # キャッシュが有効で使用可能な場合は読み込み
        if use_cache and self.is_speaker_cache_valid(enroll_dir, cache_file):
            if self.load_speaker_embeddings(cache_file):
                self.logger.info(f"キャッシュから話者データベースを読み込み: {len(self.speaker_embeddings)}名")
                return len(self.speaker_embeddings)
        
        # キャッシュが無効または使用しない場合は新規作成
        self.logger.info("話者埋め込みを新規作成します...")
        self.speaker_embeddings.clear()
        
        # 有効な話者IDを取得
        valid_speakers = self.dataset_manager.get_valid_speakers(enroll_dir)
        
        if not valid_speakers:
            self.logger.warning("登録可能な話者が見つかりません")
            return 0
        
        enrolled_count = 0
        
        for speaker_id in valid_speakers:
            speaker_path = os.path.join(enroll_dir, speaker_id)
            audio_files = self._get_audio_files(speaker_path)
            
            if self.enroll_speaker_from_files(speaker_id, audio_files):
                enrolled_count += 1
        
        # JVS話者を識別対象に含める場合、JVS埋め込みを追加
        if self.config["datasets"]["allow_jvs_speakers"]:
            jvs_count = self._add_jvs_speakers_to_database()
            if jvs_count > 0:
                enrolled_count += jvs_count
                self.logger.info(f"JVS話者を追加: {jvs_count}名")
        
        # Common Voice話者を識別対象に含める場合、Common Voice埋め込みを追加
        if self.config["datasets"]["allow_common_voice_speakers"]:
            cv_count = self._add_common_voice_speakers_to_database()
            if cv_count > 0:
                enrolled_count += cv_count
                self.logger.info(f"Common Voice話者を追加: {cv_count}名")
        
        # 埋め込みをキャッシュに保存
        if enrolled_count > 0 and auto_save:
            self.save_speaker_embeddings(cache_file)
        
        self.logger.info(f"話者データベース構築完了: {enrolled_count}名の話者を登録")
        return enrolled_count
    
    def save_speaker_embeddings(self, cache_file: str = "enrolled_speakers_embeddings.npz") -> bool:
        """
        話者埋め込みをファイルに保存
        
        Args:
            cache_file: 保存先ファイル名
            
        Returns:
            成功時True
        """
        try:
            if not self.speaker_embeddings:
                self.logger.warning("保存する埋め込みがありません")
                return False
            
            # 埋め込みデータを準備
            speaker_ids = list(self.speaker_embeddings.keys())
            embeddings = np.array(list(self.speaker_embeddings.values()))
            
            # 設定情報も保存（キャッシュ無効化チェック用）
            cache_config = {
                'allow_jvs_speakers': self.config["datasets"]["allow_jvs_speakers"],
                'allow_common_voice_speakers': self.config["datasets"]["allow_common_voice_speakers"],
                'exclude_background_speakers': self.config["datasets"]["exclude_background_speakers"]
            }
            
            # ファイルに保存
            np.savez_compressed(
                cache_file,
                embeddings=embeddings,
                speaker_ids=speaker_ids,
                cache_config=cache_config
            )
            
            self.logger.info(f"話者埋め込みを保存: {len(speaker_ids)}名の話者 -> {cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"埋め込み保存に失敗: {e}")
            return False
    
    def load_speaker_embeddings(self, cache_file: str = "enrolled_speakers_embeddings.npz") -> bool:
        """
        話者埋め込みをファイルから読み込み
        
        Args:
            cache_file: 読み込み元ファイル名
            
        Returns:
            成功時True
        """
        try:
            if not os.path.exists(cache_file):
                self.logger.info(f"埋め込みキャッシュファイルが見つかりません: {cache_file}")
                return False
            
            # ファイルから読み込み
            data = np.load(cache_file)
            embeddings = data['embeddings']
            speaker_ids = data['speaker_ids']
            
            # 設定の整合性チェック
            if 'cache_config' in data.files:
                cache_config = data['cache_config'].item()
                current_config = {
                    'allow_jvs_speakers': self.config["datasets"]["allow_jvs_speakers"],
                    'allow_common_voice_speakers': self.config["datasets"]["allow_common_voice_speakers"],
                    'exclude_background_speakers': self.config["datasets"]["exclude_background_speakers"]
                }
                
                if cache_config != current_config:
                    self.logger.info("設定が変更されているためキャッシュを無効化")
                    return False
            
            # 話者埋め込み辞書を再構築
            self.speaker_embeddings.clear()
            for speaker_id, embedding in zip(speaker_ids, embeddings):
                self.speaker_embeddings[speaker_id] = embedding
            
            self.logger.info(f"話者埋め込みを読み込み: {len(speaker_ids)}名の話者 <- {cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"埋め込み読み込みに失敗: {e}")
            return False
    
    def is_speaker_cache_valid(self, enroll_dir: str = "enroll", cache_file: str = "enrolled_speakers_embeddings.npz") -> bool:
        """
        話者キャッシュが有効かチェック
        
        Args:
            enroll_dir: 登録用ディレクトリ
            cache_file: キャッシュファイル名
            
        Returns:
            キャッシュが有効ならTrue
        """
        try:
            if not os.path.exists(cache_file):
                return False
            
            # キャッシュファイルの最終更新時刻
            cache_mtime = os.path.getmtime(cache_file)
            
            # enrollディレクトリ内の最新ファイル時刻をチェック
            latest_file_time = 0
            for root, dirs, files in os.walk(enroll_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    latest_file_time = max(latest_file_time, file_time)
            
            # キャッシュがenrollディレクトリより新しければ有効
            return cache_mtime > latest_file_time
            
        except Exception as e:
            self.logger.error(f"キャッシュ有効性チェックに失敗: {e}")
            return False
    
    def _add_jvs_speakers_to_database(self) -> int:
        """
        JVS話者を話者データベースに追加
        
        Returns:
            追加された話者数
        """
        try:
            if not self.background_loader.load_jvs_embeddings():
                self.logger.warning("JVS埋め込みの読み込みに失敗")
                return 0
            
            if self.background_loader.jvs_embeddings is None:
                return 0
                
            jvs_data = np.load("background_jvs_ecapa.npz")
            embeddings = jvs_data['embeddings']
            speaker_ids = jvs_data['speaker_ids']
            
            added_count = 0
            for speaker_id, embedding in zip(speaker_ids, embeddings):
                if speaker_id not in self.speaker_embeddings:
                    self.speaker_embeddings[speaker_id] = embedding
                    added_count += 1
            
            return added_count
            
        except Exception as e:
            self.logger.error(f"JVS話者追加に失敗: {e}")
            return 0
    
    def _add_common_voice_speakers_to_database(self) -> int:
        """
        Common Voice話者を話者データベースに追加
        
        Returns:
            追加された話者数
        """
        try:
            if not self.background_loader.load_common_voice_embeddings():
                self.logger.warning("Common Voice埋め込みの読み込みに失敗")
                return 0
            
            if self.background_loader.common_voice_embeddings is None:
                return 0
                
            cv_data = np.load("background_common_voice_ja_ecapa.npz")
            embeddings = cv_data['embeddings']
            speaker_ids = cv_data['speaker_ids']
            
            added_count = 0
            for speaker_id, embedding in zip(speaker_ids, embeddings):
                if speaker_id not in self.speaker_embeddings:
                    self.speaker_embeddings[speaker_id] = embedding
                    added_count += 1
            
            return added_count
            
        except Exception as e:
            self.logger.error(f"Common Voice話者追加に失敗: {e}")
            return 0
    
    def _get_audio_files(self, directory: str) -> List[str]:
        """ディレクトリ内の音声ファイルを取得"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        return audio_files
    
    def build_background_model(self, background_dir: str = "background_datasets") -> bool:
        """
        背景モデルの構築（スコア正規化用）
        事前計算済み埋め込みファイルを優先的に使用し、
        なければ音声ファイルから埋め込みを抽出
        
        Args:
            background_dir: 背景データセットディレクトリ
            
        Returns:
            成功時True
        """
        if not self.config["recognition"]["use_score_normalization"]:
            self.logger.info("スコア正規化が無効です")
            return True
        
        try:
            self.background_embeddings.clear()
            
            # 事前計算済み埋め込みを読み込み
            jvs_loaded, cv_loaded = self.background_loader.auto_load_all()
            
            if jvs_loaded or cv_loaded:
                # 事前計算済み埋め込みを使用
                max_samples = self.config["recognition"]["background_speakers_count"]
                combined_embeddings = self.background_loader.get_combined_background_embeddings(max_samples)
                
                if len(combined_embeddings) > 0:
                    self.background_embeddings = combined_embeddings.tolist()
                    self.logger.info(f"事前計算済み背景埋め込みを使用: {len(self.background_embeddings)}サンプル")
                else:
                    self.logger.warning("事前計算済み埋め込みが空です")
            
            # 事前計算済みが不十分な場合、音声ファイルから補完
            current_count = len(self.background_embeddings)
            target_count = self.config["recognition"]["background_speakers_count"]
            
            if current_count < target_count:
                self.logger.info(f"音声ファイルから追加埋め込み抽出: {target_count - current_count}サンプル必要")
                
                # JVSデータセット
                jvs_dir = os.path.join(background_dir, "jvs")
                if os.path.exists(jvs_dir):
                    self._process_background_dataset(jvs_dir, "JVS", target_count - len(self.background_embeddings))
                
                # Common Voiceデータセット
                cv_dir = os.path.join(background_dir, "common_voice_ja")
                if os.path.exists(cv_dir) and len(self.background_embeddings) < target_count:
                    self._process_background_dataset(cv_dir, "Common Voice", target_count - len(self.background_embeddings))
            
            if self.background_embeddings:
                # スケーラーをフィット
                background_array = np.array(self.background_embeddings)
                self.scaler.fit(background_array)
                self.logger.info(f"背景モデル構築完了: {len(self.background_embeddings)}サンプル")
                return True
            else:
                self.logger.warning("背景データが見つかりません。スコア正規化を無効にします")
                self.scaler = None
                return False
                
        except Exception as e:
            self.logger.error(f"背景モデル構築に失敗: {e}")
            return False
    
    def _process_background_dataset(self, dataset_dir: str, dataset_name: str, max_count: int):
        """背景データセットの処理"""
        count = 0
        
        for root, dirs, files in os.walk(dataset_dir):
            if count >= max_count:
                break
            
            audio_files = [f for f in files if Path(f).suffix.lower() in {'.wav', '.mp3', '.flac'}]
            
            for audio_file in audio_files[:5]:  # 各話者から最大5ファイル
                if count >= max_count:
                    break
                
                audio_path = os.path.join(root, audio_file)
                audio_tensor = self.preprocess_audio(audio_path)
                
                if audio_tensor is not None:
                    embedding = self.extract_embedding(audio_tensor)
                    if embedding is not None:
                        self.background_embeddings.append(embedding)
                        count += 1
        
        self.logger.info(f"{dataset_name}から{count}サンプルを背景モデルに追加")
    
    def recognize_speaker(self, audio_input) -> Optional[RecognitionResult]:
        """
        話者識別
        
        Args:
            audio_input: 音声ファイルパスまたは音声テンソル
            
        Returns:
            認識結果、失敗時はNone
        """
        if not self.speaker_embeddings:
            self.logger.error("話者データベースが空です")
            return None
        
        try:
            # 音声の前処理
            if isinstance(audio_input, str):
                audio_tensor = self.preprocess_audio(audio_input)
            else:
                audio_tensor = audio_input
            
            if audio_tensor is None:
                return None
            
            # 埋め込み抽出
            query_embedding = self.extract_embedding(audio_tensor)
            if query_embedding is None:
                return None
            
            # 各話者との類似度計算（背景話者は除外）
            scores = {}
            for speaker_id, speaker_embedding in self.speaker_embeddings.items():
                # 背景話者（JVS・Common Voice）は候補から除外
                if self.background_loader.should_exclude_speaker(speaker_id):
                    continue
                    
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    speaker_embedding.reshape(1, -1)
                )[0, 0]
                scores[speaker_id] = float(similarity)
            
            if not scores:
                self.logger.error("識別可能な話者がいません（全て背景話者として除外されました）")
                return None
            
            # 最も類似度の高い話者を選択
            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]
            
            # スコア正規化
            normalized_score = None
            if self.scaler is not None and self.background_embeddings:
                try:
                    # 背景埋め込みとの類似度を計算
                    bg_similarities = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        self.background_embeddings
                    )[0]
                    
                    # Z-score正規化
                    bg_mean = np.mean(bg_similarities)
                    bg_std = np.std(bg_similarities)
                    
                    if bg_std > 0:
                        normalized_score = (best_score - bg_mean) / bg_std
                    else:
                        normalized_score = best_score
                        
                except Exception as e:
                    self.logger.warning(f"スコア正規化に失敗: {e}")
                    normalized_score = best_score
            
            # 信頼度の計算
            confidence = normalized_score if normalized_score is not None else best_score
            
            # トップ10スコアを取得
            top_10_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])
            
            result = RecognitionResult(
                speaker_id=best_speaker,
                confidence=confidence,
                raw_score=best_score,
                normalized_score=normalized_score,
                all_scores=top_10_scores
            )
            
            self.logger.info(f"認識結果: {best_speaker} (信頼度: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"話者認識に失敗: {e}")
            return None
    
    def filter_scores_for_display(self, scores: Dict[str, float], show_jvs: bool = True, show_cv: bool = False) -> Dict[str, float]:
        """
        表示用にスコアをフィルタリング
        
        Args:
            scores: 全スコア辞書
            show_jvs: JVS話者を表示するか
            show_cv: Common Voice話者を表示するか
            
        Returns:
            フィルタリング済みスコア辞書（トップ10）
        """
        # まず全スコアをソート
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # フィルタリングしながらトップ10を取得
        filtered_scores = {}
        count = 0
        
        for speaker_id, score in sorted_scores:
            # 10個取得できたら終了
            if count >= 10:
                break
                
            # JVS話者の判定
            if self.dataset_manager.is_jvs_speaker(speaker_id):
                if not show_jvs:
                    continue
            
            # Common Voice話者の判定
            elif self.dataset_manager.is_common_voice_speaker(speaker_id):
                if not show_cv:
                    continue
            
            # その他の話者は常に表示
            filtered_scores[speaker_id] = score
            count += 1
        
        return filtered_scores
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        # 背景統計を取得
        bg_stats = self.background_loader.get_statistics()
        
        # 識別対象話者と除外話者を分類
        valid_speakers = []
        excluded_speakers = []
        
        for speaker_id in self.speaker_embeddings.keys():
            if self.background_loader.should_exclude_speaker(speaker_id):
                excluded_speakers.append(speaker_id)
            else:
                valid_speakers.append(speaker_id)
        
        return {
            "model_name": self.config["model"]["name"],
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "enrolled_speakers": len(self.speaker_embeddings),
            "valid_speakers": len(valid_speakers),
            "excluded_speakers": len(excluded_speakers),
            "excluded_speaker_list": excluded_speakers,
            "background_samples": len(self.background_embeddings),
            "background_stats": bg_stats,
            "score_normalization": self.scaler is not None,
            "threshold": self.threshold,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration
        }

if __name__ == "__main__":
    # テスト実行
    recognizer = JapaneseSpeakerRecognizer()
    
    if recognizer.initialize_model():
        print("モデル初期化成功")
        
        # システム情報表示
        info = recognizer.get_system_info()
        print("\n=== システム情報 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # 話者データベース構築テスト
        enrolled_count = recognizer.build_speaker_database()
        print(f"\n登録された話者数: {enrolled_count}")
        
        # 背景モデル構築テスト
        bg_success = recognizer.build_background_model()
        print(f"背景モデル構築: {'成功' if bg_success else '失敗'}")
    else:
        print("モデル初期化失敗")