"""
手動話者セグメント分離システム（リファクタリング版）
責任分離とエラーハンドリング強化版
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from segmentation_core import (
    AudioSegment, OperationResult, SegmentationConfig, Recognizer,
    AudioLoader, SegmentManager
)
from segmentation_processors import (
    LibrosaAudioProcessor, RecognitionManager, ExportManager, AudioSegmentProcessor
)
from logging_config import get_logger, setup_logging, LoggingContext
from error_handling import ErrorCategory, ErrorSeverity, error_handler_decorator


class ManualSpeakerSegmentatorV2:
    """
    手動話者セグメント分離クラス（リファクタリング版）
    
    責任分離により以下のクラスに委譲：
    - AudioLoader: 音声ファイル読み込み
    - SegmentManager: セグメント管理
    - RecognitionManager: 認識処理
    - ExportManager: エクスポート処理
    """
    
    def __init__(self, recognizer: Recognizer, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self.recognizer = recognizer
        
        # ログ設定の初期化（初回のみ）
        try:
            setup_logging()
        except:
            pass  # 既に初期化済みの場合は無視
        
        # 各専門クラスの初期化
        self.audio_processor = LibrosaAudioProcessor()
        self.audio_loader = AudioLoader(self.audio_processor, self.config)
        self.segment_manager = SegmentManager(self.config)
        self.recognition_manager = RecognitionManager(recognizer, self.config)
        self.export_manager = ExportManager(self.config)
        self.segment_processor = AudioSegmentProcessor(self.config)
        
        # 音声データ
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.audio_duration: Optional[float] = None
        
        # ログ設定
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("ManualSpeakerSegmentatorV2初期化完了")
    
    # === 音声ファイル管理 ===
    
    def load_audio(self, audio_file_path: str) -> OperationResult:
        """音声ファイルの読み込み"""
        result = self.audio_loader.load_audio_file(audio_file_path)
        
        if result.success:
            self.audio_data = result.data['audio_data']
            self.sample_rate = result.data['sample_rate']
            self.audio_duration = result.data['duration']
            
            # セグメント情報をリセット
            self.segment_manager.clear_segments()
            self.segment_processor.clear_cache()
        
        return result
    
    def get_waveform_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """波形表示用のデータを取得"""
        if self.audio_data is None:
            raise ValueError("音声データが読み込まれていません")
        
        # 時間軸を生成
        time_axis = np.linspace(0, self.audio_duration, len(self.audio_data))
        return time_axis, self.audio_data, self.sample_rate
    
    def save_audio_for_playback(self, output_path: str) -> OperationResult:
        """再生用音声ファイルの保存"""
        try:
            if self.audio_data is None:
                return OperationResult(
                    success=False,
                    message="音声データが読み込まれていません"
                )
            
            self.audio_processor.save_audio(self.audio_data, output_path, self.sample_rate)
            return OperationResult(
                success=True,
                message=f"再生用ファイル保存完了: {output_path}"
            )
            
        except Exception as e:
            self.logger.exception("再生用ファイル保存エラー")
            return OperationResult(
                success=False,
                message=f"再生用ファイル保存エラー: {str(e)}"
            )
    
    def create_segment_audio_file(self, segment: AudioSegment, output_path: str) -> OperationResult:
        """セグメント音声ファイルの作成"""
        try:
            if self.audio_data is None:
                return OperationResult(
                    success=False,
                    message="音声データが読み込まれていません"
                )
            
            segment_audio = self.segment_processor.extract_segment_audio(
                segment, self.audio_data, self.sample_rate
            )
            self.audio_processor.save_audio(segment_audio, output_path, self.sample_rate)
            
            return OperationResult(
                success=True,
                message=f"セグメント音声ファイル作成完了: {output_path}"
            )
            
        except Exception as e:
            self.logger.exception("セグメント音声ファイル作成エラー")
            return OperationResult(
                success=False,
                message=f"セグメント音声ファイル作成エラー: {str(e)}"
            )
    
    # === セグメント管理 ===
    
    def create_segment(self, start_time: float, end_time: float) -> OperationResult:
        """セグメントの作成"""
        if self.audio_duration is None:
            return OperationResult(
                success=False,
                message="音声データが読み込まれていません"
            )
        
        return self.segment_manager.create_segment(start_time, end_time, self.audio_duration)
    
    def delete_segment(self, segment_id: int) -> OperationResult:
        """セグメントの削除"""
        return self.segment_manager.delete_segment(segment_id)
    
    def update_segment_times(self, segment_id: int, start_time: float, end_time: float) -> OperationResult:
        """セグメント時間の更新"""
        if self.audio_duration is None:
            return OperationResult(
                success=False,
                message="音声データが読み込まれていません"
            )
        
        return self.segment_manager.update_segment_times(
            segment_id, start_time, end_time, self.audio_duration
        )
    
    def get_segments_list(self) -> List[AudioSegment]:
        """セグメント一覧を取得（時間順）"""
        return self.segment_manager.get_segments_list()
    
    def assign_speaker_to_segment(self, segment_id: int, speaker_name: str) -> OperationResult:
        """セグメントに話者を割り当て"""
        return self.segment_manager.assign_speaker_to_segment(segment_id, speaker_name)
    
    # === 認識処理 ===
    
    def recognize_all_segments(self, show_jvs: bool = False, show_cv: bool = False) -> OperationResult:
        """全セグメントの認識実行（同期処理）"""
        if self.audio_data is None:
            return OperationResult(
                success=False,
                message="音声データが読み込まれていません"
            )
        
        with LoggingContext(self.__class__.__name__, "全セグメント認識"):
            return self.recognition_manager.recognize_all_segments(
                self.segment_manager.segments, self.audio_data, self.sample_rate, show_jvs, show_cv
            )
    
    async def recognize_all_segments_async(self, show_jvs: bool = False, show_cv: bool = False) -> OperationResult:
        """全セグメントの認識実行（非同期処理）"""
        if self.audio_data is None:
            return OperationResult(
                success=False,
                message="音声データが読み込まれていません"
            )
        
        return await self.recognition_manager.recognize_all_segments_async(
            self.segment_manager.segments, self.audio_data, self.sample_rate, show_jvs, show_cv
        )
    
    def recognize_segment(self, segment_id: int, show_jvs: bool = False, show_cv: bool = False) -> OperationResult:
        """単一セグメントの認識"""
        if self.audio_data is None:
            return OperationResult(
                success=False,
                message="音声データが読み込まれていません"
            )
        
        if segment_id not in self.segment_manager.segments:
            return OperationResult(
                success=False,
                message=f"セグメント{segment_id}が見つかりません"
            )
        
        segment = self.segment_manager.segments[segment_id]
        return self.recognition_manager.recognize_segment(
            segment, self.audio_data, self.sample_rate, show_jvs, show_cv
        )
    
    # === データ取得・分析 ===
    
    def get_timeline_data(self) -> Dict[str, Any]:
        """タイムライン表示用のデータを生成"""
        timeline_data = {
            'speakers': {},
            'total_duration': self.audio_duration or 0.0,
            'segments': []
        }
        
        # 話者別にセグメントを分類
        for segment in self.segment_manager.segments.values():
            speaker = segment.assigned_speaker or segment.top_speaker or 'unknown'
            
            if speaker not in timeline_data['speakers']:
                timeline_data['speakers'][speaker] = {
                    'segments': [],
                    'total_time': 0.0,
                    'segment_count': 0
                }
            
            timeline_data['speakers'][speaker]['segments'].append({
                'start': segment.start_time,
                'end': segment.end_time,
                'duration': segment.duration,
                'confidence': segment.top_score
            })
            timeline_data['speakers'][speaker]['total_time'] += segment.duration
            timeline_data['speakers'][speaker]['segment_count'] += 1
        
        # 全セグメント情報
        for segment in self.get_segments_list():
            timeline_data['segments'].append({
                'id': segment.id,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'assigned_speaker': segment.assigned_speaker,
                'top_speaker': segment.top_speaker,
                'top_score': segment.top_score,
                'confidence_level': segment.confidence_level
            })
        
        return timeline_data
    
    # === エクスポート機能 ===
    
    def export_to_csv(self, output_path: str) -> OperationResult:
        """CSV形式でエクスポート"""
        segments = self.get_segments_list()
        return self.export_manager.export_to_csv(segments, output_path)
    
    def export_to_json(self, output_path: str) -> OperationResult:
        """JSON形式でエクスポート"""
        timeline_data = self.get_timeline_data()
        return self.export_manager.export_to_json(timeline_data, output_path)
    
    def export_to_srt(self, output_path: str) -> OperationResult:
        """SRT字幕形式でエクスポート"""
        segments = self.get_segments_list()
        return self.export_manager.export_to_srt(segments, output_path)
    
    # === 設定・状態管理 ===
    
    def get_config(self) -> SegmentationConfig:
        """現在の設定を取得"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """設定を更新"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"設定更新: {key} = {value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        segments = self.get_segments_list()
        
        if not segments:
            return {
                'total_segments': 0,
                'total_duration': 0.0,
                'recognized_segments': 0,
                'avg_segment_duration': 0.0,
                'confidence_distribution': {}
            }
        
        recognized_segments = [s for s in segments if s.is_recognized]
        total_duration = sum(s.duration for s in segments)
        avg_duration = total_duration / len(segments)
        
        confidence_dist = {}
        for segment in recognized_segments:
            level = segment.confidence_level or 'unknown'
            confidence_dist[level] = confidence_dist.get(level, 0) + 1
        
        return {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'recognized_segments': len(recognized_segments),
            'avg_segment_duration': avg_duration,
            'confidence_distribution': confidence_dist,
            'audio_duration': self.audio_duration or 0.0,
            'coverage_ratio': (total_duration / (self.audio_duration or 1.0)) * 100
        }
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self.segment_processor.clear_cache()
        self.segment_manager.clear_segments()
        self.audio_data = None
        self.sample_rate = None
        self.audio_duration = None
        self.logger.info("リソースクリーンアップ完了")