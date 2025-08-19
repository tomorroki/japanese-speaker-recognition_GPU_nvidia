"""
セグメンテーション機能のコア定義
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Protocol
from type_definitions import (
    AudioData, SampleRate, TimeStamp, Duration, FilePath, 
    SegmentID, SpeakerName, AudioProcessor as AudioProcessorProtocol,
    Recognizer as RecognizerProtocol, Validatable, TypeValidationError
)
from abc import ABC, abstractmethod
import logging
import numpy as np
# Avoid circular imports by importing these dynamically when needed
# from error_handling import ErrorCategory, ErrorSeverity, error_handler_decorator
# from logging_config import get_logger


# エラーコード定義
class ErrorCode(Enum):
    AUDIO_LOAD_FAILED = "AUDIO_LOAD_FAILED"
    SEGMENT_OVERLAP = "SEGMENT_OVERLAP"
    INVALID_TIME_RANGE = "INVALID_TIME_RANGE"
    RECOGNITION_FAILED = "RECOGNITION_FAILED"
    SEGMENT_NOT_FOUND = "SEGMENT_NOT_FOUND"
    EXPORT_FAILED = "EXPORT_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# 操作結果の型安全な管理
@dataclass
class OperationResult:
    """操作結果を型安全に管理"""
    success: bool
    message: str
    error_code: Optional[ErrorCode] = None
    data: Optional[Any] = None


# セグメント情報データクラス
@dataclass
class AudioSegment(Validatable):
    """音声セグメント情報"""
    id: SegmentID
    start_time: TimeStamp
    end_time: TimeStamp
    duration: Duration
    assigned_speaker: Optional[SpeakerName] = None
    recognition_results: Optional[List[Dict[str, Any]]] = None
    confidence_level: Optional[str] = None  # "高", "中", "低"
    
    @property
    def is_recognized(self) -> bool:
        """認識済みかどうか"""
        return self.recognition_results is not None
    
    @property
    def top_speaker(self) -> Optional[str]:
        """最上位話者候補"""
        if self.recognition_results and len(self.recognition_results) > 0:
            return self.recognition_results[0].get('speaker')
        return None
    
    @property
    def top_score(self) -> Optional[float]:
        """最上位スコア"""
        if self.recognition_results and len(self.recognition_results) > 0:
            return self.recognition_results[0].get('score')
        return None
    
    def validate(self) -> List[str]:
        """セグメント検証"""
        errors = []
        
        if self.start_time >= self.end_time:
            errors.append("開始時間は終了時間より前である必要があります")
        
        if self.start_time < 0:
            errors.append("開始時間は0以上である必要があります")
        
        if self.duration != (self.end_time - self.start_time):
            errors.append("duration が時間範囲と一致しません")
        
        if self.duration <= 0:
            errors.append("duration は正の値である必要があります")
        
        return errors
    
    def is_valid(self) -> bool:
        """有効性チェック"""
        return len(self.validate()) == 0


# 設定管理の集約
@dataclass
class SegmentationConfig:
    """セグメンテーション設定を一元管理"""
    min_segment_duration: float = 0.5
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.0
    })
    max_recognition_results: int = 5
    temp_file_suffix: str = '.wav'
    max_cache_size: int = 100
    max_parallel_workers: int = 4


# プロトコル定義
class AudioProcessor(AudioProcessorProtocol):
    """音声処理のプロトコル"""
    
    def load_audio(self, file_path: FilePath) -> Tuple[AudioData, SampleRate]:
        """音声ファイルを読み込む"""
        ...
    
    def save_audio(self, audio_data: AudioData, file_path: FilePath, sample_rate: SampleRate) -> None:
        """音声データをファイルに保存"""
        ...


class Recognizer(RecognizerProtocol):
    """認識器のプロトコル"""
    
    def recognize_speaker(self, audio_path: FilePath) -> Optional[Any]:
        """話者認識を実行"""
        ...
    
    @property
    def speaker_embeddings(self) -> Dict[SpeakerName, Any]:
        """話者エンベディング取得"""
        ...


# 抽象基底クラス
class SegmentValidator:
    """セグメント検証ロジック"""
    
    @staticmethod
    def validate_time_range(start: float, end: float, max_duration: float, 
                          min_duration: float = 0.5) -> Optional[str]:
        """時間範囲の検証"""
        if start >= end:
            return "開始時間は終了時間より前である必要があります"
        if start < 0 or end > max_duration:
            return f"時間は0秒から{max_duration:.2f}秒の間で指定してください"
        if end - start < min_duration:
            return f"セグメントは最低{min_duration}秒以上である必要があります"
        return None
    
    @staticmethod
    def validate_segment_overlap(start: float, end: float, existing_segments: List[AudioSegment],
                               exclude_id: Optional[int] = None) -> Optional[str]:
        """セグメント重複の検証"""
        for segment in existing_segments:
            if exclude_id and segment.id == exclude_id:
                continue
            
            if not (end <= segment.start_time or start >= segment.end_time):
                return f"セグメント{segment.id}と時間が重複しています"
        return None


class AudioLoader:
    """音声ファイル読み込み専用クラス"""
    
    def __init__(self, processor: AudioProcessor, config: SegmentationConfig):
        self.processor = processor
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_audio_file(self, file_path: str) -> OperationResult:
        """音声ファイルの読み込み"""
        try:
            self.logger.info(f"音声ファイル読み込み開始: {file_path}")
            audio_data, sample_rate = self.processor.load_audio(file_path)
            duration = len(audio_data) / sample_rate
            
            result_data = {
                'audio_data': audio_data,
                'sample_rate': sample_rate,
                'duration': duration
            }
            
            self.logger.info(f"音声読み込み完了: {duration:.2f}秒")
            return OperationResult(
                success=True,
                message=f"音声読み込み完了: {duration:.2f}秒",
                data=result_data
            )
            
        except Exception as e:
            self.logger.exception("音声読み込みエラー")
            return OperationResult(
                success=False,
                message=f"音声読み込みエラー: {str(e)}",
                error_code=ErrorCode.AUDIO_LOAD_FAILED
            )


class SegmentManager:
    """セグメント管理専用クラス"""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.segments: Dict[int, AudioSegment] = {}
        self.next_segment_id = 1
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_segment(self, start_time: float, end_time: float, 
                      audio_duration: float) -> OperationResult:
        """セグメントの作成"""
        try:
            self.logger.info(f"セグメント作成開始: {start_time:.1f}s-{end_time:.1f}s")
            
            # 時間範囲の検証
            time_validation = SegmentValidator.validate_time_range(
                start_time, end_time, audio_duration, self.config.min_segment_duration
            )
            if time_validation:
                return OperationResult(
                    success=False,
                    message=time_validation,
                    error_code=ErrorCode.INVALID_TIME_RANGE
                )
            
            # 重複検証
            overlap_validation = SegmentValidator.validate_segment_overlap(
                start_time, end_time, list(self.segments.values())
            )
            if overlap_validation:
                return OperationResult(
                    success=False,
                    message=overlap_validation,
                    error_code=ErrorCode.SEGMENT_OVERLAP
                )
            
            # セグメント作成
            duration = end_time - start_time
            segment = AudioSegment(
                id=self.next_segment_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
            self.segments[self.next_segment_id] = segment
            self.next_segment_id += 1
            
            message = f"セグメント{segment.id}を作成しました ({start_time:.1f}s-{end_time:.1f}s)"
            self.logger.info(message)
            
            return OperationResult(
                success=True,
                message=message,
                data=segment
            )
            
        except Exception as e:
            self.logger.exception("セグメント作成エラー")
            return OperationResult(
                success=False,
                message=f"セグメント作成エラー: {str(e)}",
                error_code=ErrorCode.VALIDATION_ERROR
            )
    
    def delete_segment(self, segment_id: int) -> OperationResult:
        """セグメントの削除"""
        if segment_id not in self.segments:
            return OperationResult(
                success=False,
                message=f"セグメント{segment_id}が見つかりません",
                error_code=ErrorCode.SEGMENT_NOT_FOUND
            )
        
        del self.segments[segment_id]
        message = f"セグメント{segment_id}を削除しました"
        self.logger.info(message)
        
        return OperationResult(success=True, message=message)
    
    def update_segment_times(self, segment_id: int, start_time: float, 
                           end_time: float, audio_duration: float) -> OperationResult:
        """セグメント時間の更新"""
        if segment_id not in self.segments:
            return OperationResult(
                success=False,
                message=f"セグメント{segment_id}が見つかりません",
                error_code=ErrorCode.SEGMENT_NOT_FOUND
            )
        
        # 検証（対象セグメントを除く）
        time_validation = SegmentValidator.validate_time_range(
            start_time, end_time, audio_duration, self.config.min_segment_duration
        )
        if time_validation:
            return OperationResult(
                success=False,
                message=time_validation,
                error_code=ErrorCode.INVALID_TIME_RANGE
            )
        
        overlap_validation = SegmentValidator.validate_segment_overlap(
            start_time, end_time, list(self.segments.values()), exclude_id=segment_id
        )
        if overlap_validation:
            return OperationResult(
                success=False,
                message=overlap_validation,
                error_code=ErrorCode.SEGMENT_OVERLAP
            )
        
        # 更新
        segment = self.segments[segment_id]
        segment.start_time = start_time
        segment.end_time = end_time
        segment.duration = end_time - start_time
        
        message = f"セグメント{segment_id}を更新しました"
        self.logger.info(message)
        
        return OperationResult(success=True, message=message)
    
    def get_segments_list(self) -> List[AudioSegment]:
        """セグメント一覧を取得（時間順）"""
        return sorted(self.segments.values(), key=lambda x: x.start_time)
    
    def clear_segments(self) -> None:
        """すべてのセグメントをクリア"""
        self.segments.clear()
        self.next_segment_id = 1
        self.logger.info("全セグメントをクリアしました")
    
    def assign_speaker_to_segment(self, segment_id: int, speaker_name: str) -> OperationResult:
        """セグメントに話者を割り当て"""
        if segment_id not in self.segments:
            return OperationResult(
                success=False,
                message=f"セグメント{segment_id}が見つかりません",
                error_code=ErrorCode.SEGMENT_NOT_FOUND
            )
        
        self.segments[segment_id].assigned_speaker = speaker_name
        message = f"セグメント{segment_id}に話者'{speaker_name}'を割り当てました"
        self.logger.info(message)
        
        return OperationResult(success=True, message=message)