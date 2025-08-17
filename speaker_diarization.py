"""
複数話者分析専用モジュール
既存システムと完全分離
"""

import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# pyannote.audioのインポート
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from segment_processor import SegmentProcessor
from enhanced_speaker_recognition import JapaneseSpeakerRecognizer

# 環境変数読み込み
load_dotenv()


@dataclass
class DiarizationSegment:
    """ダイアライゼーションセグメント情報"""
    start: float
    end: float
    speaker_label: str
    duration: float


@dataclass
class DiarizationResult:
    """ダイアライゼーション結果"""
    segments: List[Dict[str, Any]]
    total_speakers: int
    total_duration: float
    file_info: Dict[str, Any]


class SpeakerDiarizer:
    """pyannote.audio ダイアライゼーション処理"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.pipeline = None
        self.device = self._setup_device()
        self.logger = logging.getLogger(__name__)
        
        if not PYANNOTE_AVAILABLE:
            self.logger.error("pyannote.audioがインストールされていません")
    
    def _setup_device(self) -> torch.device:
        """デバイス設定"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def initialize(self) -> bool:
        """
        モデル初期化
        
        Returns:
            初期化成功の可否
        """
        if not PYANNOTE_AVAILABLE:
            self.logger.error("pyannote.audioが利用できません")
            return False
        
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token or hf_token == "your_huggingface_token_here":
                self.logger.error("Hugging Face tokenが設定されていません")
                return False
            
            self.logger.info("pyannote.audioモデルを初期化中...")
            self.pipeline = Pipeline.from_pretrained(
                self.config["diarization"]["model"],
                use_auth_token=hf_token
            )
            self.pipeline.to(self.device)
            self.logger.info(f"ダイアライゼーションモデル初期化完了 (デバイス: {self.device})")
            return True
            
        except Exception as e:
            self.logger.error(f"ダイアライゼーションモデル初期化エラー: {e}")
            return False
    
    def diarize(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 10) -> List[DiarizationSegment]:
        """
        音声ファイルのダイアライゼーション
        元音声をそのまま処理
        
        Args:
            audio_path: 音声ファイルパス
            min_speakers: 最小話者数
            max_speakers: 最大話者数
            
        Returns:
            ダイアライゼーションセグメントリスト
        """
        if self.pipeline is None:
            raise RuntimeError("モデルが初期化されていません")
        
        try:
            self.logger.info(f"ダイアライゼーション開始: {audio_path}")
            
            # pyannote.audioで処理（前処理なし）
            diarization = self.pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # セグメント情報抽出
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # 最小セグメント長フィルタ
                duration = turn.end - turn.start
                if duration >= self.config["diarization"]["min_segment_duration"]:
                    segments.append(DiarizationSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker_label=speaker,
                        duration=duration
                    ))
            
            self.logger.info(f"ダイアライゼーション完了: {len(segments)}セグメント検出")
            return segments
            
        except Exception as e:
            self.logger.error(f"ダイアライゼーションエラー: {e}")
            raise


class MultiSpeakerRecognizer:
    """複数話者分析・認識の統合処理"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.diarizer = SpeakerDiarizer(self.config)
        self.segment_processor = SegmentProcessor(
            target_sr=self.config["segment_processing"]["target_sample_rate"]
        )
        self.recognizer = None  # 遅延初期化
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定読み込み
        
        Args:
            config_path: 設定ファイルパス
            
        Returns:
            設定辞書
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"設定ファイル読み込みエラー: {e}")
    
    def initialize(self) -> bool:
        """
        システム初期化
        
        Returns:
            初期化成功の可否
        """
        try:
            # ダイアライゼーションモデル
            if not self.diarizer.initialize():
                return False
            
            # 話者認識モデル（既存システム利用）
            self.logger.info("話者認識システムを初期化中...")
            self.recognizer = JapaneseSpeakerRecognizer()
            if not self.recognizer.initialize_model():
                self.logger.error("話者認識モデル初期化失敗")
                return False
            
            # 話者データベース構築
            enrolled_count = self.recognizer.build_speaker_database()
            self.logger.info(f"話者認識システム初期化完了: {enrolled_count}名登録")
            
            return True
            
        except Exception as e:
            self.logger.error(f"システム初期化エラー: {e}")
            return False
    
    def process_audio(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 10) -> DiarizationResult:
        """
        完全な複数話者分析処理
        
        Args:
            audio_path: 音声ファイルパス
            min_speakers: 最小話者数
            max_speakers: 最大話者数
            
        Returns:
            ダイアライゼーション結果
        """
        if self.recognizer is None:
            raise RuntimeError("システムが初期化されていません")
        
        try:
            # 1. ダイアライゼーション（元音声そのまま）
            self.logger.info("ダイアライゼーション実行中...")
            diarization_segments = self.diarizer.diarize(audio_path, min_speakers, max_speakers)
            
            # 2. 各セグメントの話者認識
            self.logger.info("セグメント別話者認識実行中...")
            results = []
            
            for i, segment in enumerate(diarization_segments):
                try:
                    # セグメント切り出し・前処理（認識用）
                    audio_chunk = self.segment_processor.extract_segment(
                        audio_path, segment.start, segment.end
                    )
                    
                    # セグメント有効性チェック
                    if not self.segment_processor.validate_segment(audio_chunk):
                        self.logger.warning(f"無効なセグメント {i+1}: {segment.start:.1f}s-{segment.end:.1f}s")
                        continue
                    
                    # 話者認識実行
                    recognition = self.recognizer.recognize_segment(audio_chunk)
                    
                    results.append({
                        'segment_id': i + 1,
                        'start_time': segment.start,
                        'end_time': segment.end,
                        'duration': segment.duration,
                        'diarization_label': segment.speaker_label,
                        'recognized_speaker': recognition.speaker_id if recognition else "未認識",
                        'confidence': recognition.confidence if recognition else 0.0,
                        'raw_score': recognition.raw_score if recognition else 0.0
                    })
                    
                except Exception as e:
                    self.logger.error(f"セグメント {i+1} 処理エラー: {e}")
                    continue
            
            # 結果構築
            total_duration = max([s.end for s in diarization_segments]) if diarization_segments else 0.0
            unique_speakers = len(set(s.speaker_label for s in diarization_segments))
            
            return DiarizationResult(
                segments=results,
                total_speakers=unique_speakers,
                total_duration=total_duration,
                file_info={
                    'path': audio_path,
                    'total_segments': len(results),
                    'processed_segments': len([r for r in results if r['recognized_speaker'] != "未認識"])
                }
            )
            
        except Exception as e:
            self.logger.error(f"音声処理エラー: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        システム情報取得
        
        Returns:
            システム情報辞書
        """
        info = {
            "diarization_model": self.config["diarization"]["model"],
            "diarization_device": str(self.diarizer.device),
            "target_sample_rate": self.config["segment_processing"]["target_sample_rate"],
            "pyannote_available": PYANNOTE_AVAILABLE,
            "diarizer_initialized": self.diarizer.pipeline is not None,
            "recognizer_initialized": self.recognizer is not None
        }
        
        if self.recognizer:
            recognition_info = self.recognizer.get_system_info()
            info.update({
                "enrolled_speakers": recognition_info["enrolled_speakers"],
                "recognition_device": recognition_info["device"]
            })
        
        return info