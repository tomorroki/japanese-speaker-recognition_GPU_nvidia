"""
セグメンテーション処理の専門クラス
"""

import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import json
import csv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any
import logging
from functools import wraps

from segmentation_core import (
    AudioSegment, OperationResult, ErrorCode, SegmentationConfig,
    AudioProcessor, Recognizer
)
from error_handling import (
    ErrorCategory, ErrorSeverity, error_handler_decorator, 
    retry_with_backoff, AudioProcessingError, RecognitionError
)
# Import dynamically to avoid circular dependencies
# from logging_config import get_logger, LoggingContext
# from performance_optimizer import (
#     get_memory_manager, get_profiler, profile_performance, 
#     memory_optimized, create_cache
# )


# ログデコレータ
def log_operation(operation_name: str):
    """操作ログデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = logging.getLogger(self.__class__.__name__)
            logger.info(f"{operation_name}開始")
            
            try:
                result = func(self, *args, **kwargs)
                success = result.success if hasattr(result, 'success') else result[0]
                logger.info(f"{operation_name}完了: 成功={success}")
                return result
            except Exception as e:
                logger.exception(f"{operation_name}エラー")
                raise
        return wrapper
    return decorator


# コンテキストマネージャー
@contextmanager
def temporary_audio_file(audio_data: np.ndarray, sample_rate: int, suffix: str = '.wav'):
    """一時音声ファイルのコンテキストマネージャ"""
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_file.close()
        yield temp_file.name
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


# librosa実装
class LibrosaAudioProcessor(AudioProcessor):
    """librosaを使った音声処理実装"""
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """音声ファイルの読み込み"""
        return librosa.load(file_path, sr=None, mono=True)
    
    def save_audio(self, audio_data: np.ndarray, file_path: str, sample_rate: int) -> None:
        """音声データのファイル保存"""
        sf.write(file_path, audio_data, sample_rate)


# セグメント音声処理
class AudioSegmentProcessor:
    """音声セグメント処理の効率化"""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._audio_cache = {}  # Simplified cache for now
        self.logger = logging.getLogger(self.__class__.__name__)
        # get_memory_manager().register_for_cleanup(self)  # Disabled to avoid import issues
    
    def extract_segment_audio(self, segment: AudioSegment, audio_data: np.ndarray, 
                            sample_rate: int) -> np.ndarray:
        """セグメント音声の抽出"""
        start_sample = int(segment.start_time * sample_rate)
        end_sample = int(segment.end_time * sample_rate)
        return audio_data[start_sample:end_sample]
    
    def get_cached_segment_audio(self, segment: AudioSegment, audio_data: np.ndarray, 
                               sample_rate: int) -> np.ndarray:
        """キャッシュ機能付きセグメント音声取得"""
        cache_key = f"{segment.id}_{segment.start_time}_{segment.end_time}"
        
        # キャッシュから取得試行
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # キャッシュミス - セグメント音声抽出
        segment_audio = self.extract_segment_audio(segment, audio_data, sample_rate)
        
        # キャッシュに保存（サイズ制限付き）
        if len(self._audio_cache) < 100:
            self._audio_cache[cache_key] = segment_audio
        
        return segment_audio
    
    def clear_cache(self) -> None:
        """キャッシュクリア"""
        self._audio_cache.clear()
        self.logger.info("音声キャッシュをクリアしました")


class RecognitionManager:
    """認識処理専用クラス"""
    
    def __init__(self, recognizer: Recognizer, config: SegmentationConfig):
        self.recognizer = recognizer
        self.config = config
        self.segment_processor = AudioSegmentProcessor(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def recognize_segment(self, segment: AudioSegment, audio_data: np.ndarray,
                         sample_rate: int, show_jvs: bool = False, 
                         show_cv: bool = False) -> OperationResult:
        """単一セグメントの認識"""
        try:
            # セグメント音声抽出
            segment_audio = self.segment_processor.extract_segment_audio(
                segment, audio_data, sample_rate
            )
            
            # 一時ファイルとして保存して認識
            with temporary_audio_file(segment_audio, sample_rate, self.config.temp_file_suffix) as temp_path:
                result = self.recognizer.recognize_speaker(temp_path)
                
                if result and result.all_scores:
                    # 結果をJVS/Common Voice設定に基づいてフィルタリング
                    filtered_scores = self._filter_recognition_results(
                        result.all_scores, show_jvs, show_cv
                    )
                    
                    # スコア順でソート
                    sorted_results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top-N結果を保存
                    recognition_results = []
                    max_results = self.config.max_recognition_results
                    for i, (speaker, score) in enumerate(sorted_results[:max_results]):
                        recognition_results.append({
                            'rank': i + 1,
                            'speaker': speaker,
                            'score': float(score)
                        })
                    
                    # セグメントに結果を設定
                    segment.recognition_results = recognition_results
                    
                    # 信頼度レベル設定
                    top_score = sorted_results[0][1] if sorted_results else 0.0
                    segment.confidence_level = self._determine_confidence_level(top_score)
                    
                    return OperationResult(
                        success=True,
                        message=f"セグメント{segment.id}を認識しました",
                        data=segment
                    )
                else:
                    return OperationResult(
                        success=False,
                        message=f"セグメント{segment.id}の認識に失敗しました",
                        error_code=ErrorCode.RECOGNITION_FAILED
                    )
        
        except Exception as e:
            self.logger.exception(f"セグメント{segment.id}認識エラー")
            return OperationResult(
                success=False,
                message=f"認識エラー: {str(e)}",
                error_code=ErrorCode.RECOGNITION_FAILED
            )
    
    @log_operation("全セグメント認識")
    def recognize_all_segments(self, segments: Dict[int, AudioSegment], 
                             audio_data: np.ndarray, sample_rate: int,
                             show_jvs: bool = False, show_cv: bool = False) -> OperationResult:
        """全セグメントの同期認識"""
        if not segments:
            return OperationResult(
                success=False,
                message="認識するセグメントがありません"
            )
        
        recognized_count = 0
        total_count = len(segments)
        
        for segment in segments.values():
            result = self.recognize_segment(segment, audio_data, sample_rate, show_jvs, show_cv)
            if result.success:
                recognized_count += 1
        
        return OperationResult(
            success=True,
            message=f"{recognized_count}/{total_count}個のセグメントを認識しました"
        )
    
    async def recognize_all_segments_async(self, segments: Dict[int, AudioSegment],
                                         audio_data: np.ndarray, sample_rate: int,
                                         show_jvs: bool = False, show_cv: bool = False) -> OperationResult:
        """全セグメントの非同期認識"""
        if not segments:
            return OperationResult(
                success=False,
                message="認識するセグメントがありません"
            )
        
        # セグメント音声を事前に抽出（並列処理で効率化）
        segment_audio_pairs = []
        for segment in segments.values():
            segment_audio = self.segment_processor.get_cached_segment_audio(
                segment, audio_data, sample_rate
            )
            segment_audio_pairs.append((segment, segment_audio))
        
        # 並列認識実行
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            tasks = [
                self._recognize_segment_async(executor, segment, segment_audio, sample_rate, show_jvs, show_cv)
                for segment, segment_audio in segment_audio_pairs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果処理
        recognized_count = sum(1 for r in results if isinstance(r, OperationResult) and r.success)
        total_count = len(segments)
        
        return OperationResult(
            success=True,
            message=f"{recognized_count}/{total_count}個のセグメントを並列認識しました"
        )
    
    async def _recognize_segment_async(self, executor: ThreadPoolExecutor, segment: AudioSegment,
                                     segment_audio: np.ndarray, sample_rate: int,
                                     show_jvs: bool, show_cv: bool) -> OperationResult:
        """非同期セグメント認識"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, self._recognize_segment_sync, segment, segment_audio, sample_rate, show_jvs, show_cv
        )
    
    def _recognize_segment_sync(self, segment: AudioSegment, segment_audio: np.ndarray,
                              sample_rate: int, show_jvs: bool, show_cv: bool) -> OperationResult:
        """同期セグメント認識（並列処理用）"""
        try:
            with temporary_audio_file(segment_audio, sample_rate, self.config.temp_file_suffix) as temp_path:
                result = self.recognizer.recognize_speaker(temp_path)
                
                if result and result.all_scores:
                    filtered_scores = self._filter_recognition_results(
                        result.all_scores, show_jvs, show_cv
                    )
                    
                    sorted_results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    recognition_results = []
                    max_results = self.config.max_recognition_results
                    for i, (speaker, score) in enumerate(sorted_results[:max_results]):
                        recognition_results.append({
                            'rank': i + 1,
                            'speaker': speaker,
                            'score': float(score)
                        })
                    
                    segment.recognition_results = recognition_results
                    top_score = sorted_results[0][1] if sorted_results else 0.0
                    segment.confidence_level = self._determine_confidence_level(top_score)
                    
                    return OperationResult(success=True, message="認識成功", data=segment)
                else:
                    return OperationResult(
                        success=False, 
                        message="認識失敗", 
                        error_code=ErrorCode.RECOGNITION_FAILED
                    )
        except Exception as e:
            return OperationResult(
                success=False,
                message=f"認識エラー: {str(e)}",
                error_code=ErrorCode.RECOGNITION_FAILED
            )
    
    def _filter_recognition_results(self, all_scores: Dict[str, float], 
                                  show_jvs: bool, show_cv: bool) -> Dict[str, float]:
        """認識結果をJVS/Common Voice設定に基づいてフィルタリング"""
        filtered_scores = {}
        
        for speaker, score in all_scores.items():
            is_jvs = speaker.startswith('jvs')
            is_cv = speaker.startswith(('cv_', 'commonvoice_'))
            
            should_include = True
            if is_jvs and not show_jvs:
                should_include = False
            elif is_cv and not show_cv:
                should_include = False
            
            if should_include:
                filtered_scores[speaker] = score
        
        return filtered_scores
    
    def _determine_confidence_level(self, score: float) -> str:
        """スコアから信頼度レベルを決定"""
        thresholds = self.config.confidence_thresholds
        if score >= thresholds['high']:
            return "高"
        elif score >= thresholds['medium']:
            return "中"
        else:
            return "低"


class ExportManager:
    """エクスポート処理専用クラス"""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @log_operation("CSV エクスポート")
    def export_to_csv(self, segments: List[AudioSegment], output_path: str) -> OperationResult:
        """CSV形式でエクスポート"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['segment_id', 'start_time', 'end_time', 'duration', 
                             'assigned_speaker', 'top_speaker', 'top_score', 'confidence_level']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for segment in segments:
                    writer.writerow({
                        'segment_id': segment.id,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'duration': segment.duration,
                        'assigned_speaker': segment.assigned_speaker or '',
                        'top_speaker': segment.top_speaker or '',
                        'top_score': segment.top_score or '',
                        'confidence_level': segment.confidence_level or ''
                    })
            
            return OperationResult(
                success=True,
                message=f"CSVファイルを保存しました: {output_path}"
            )
            
        except Exception as e:
            self.logger.exception("CSV保存エラー")
            return OperationResult(
                success=False,
                message=f"CSV保存エラー: {str(e)}",
                error_code=ErrorCode.EXPORT_FAILED
            )
    
    @log_operation("JSON エクスポート")
    def export_to_json(self, timeline_data: Dict[str, Any], output_path: str) -> OperationResult:
        """JSON形式でエクスポート"""
        try:
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(timeline_data, jsonfile, indent=2, ensure_ascii=False)
            
            return OperationResult(
                success=True,
                message=f"JSONファイルを保存しました: {output_path}"
            )
            
        except Exception as e:
            self.logger.exception("JSON保存エラー")
            return OperationResult(
                success=False,
                message=f"JSON保存エラー: {str(e)}",
                error_code=ErrorCode.EXPORT_FAILED
            )
    
    @log_operation("SRT エクスポート")
    def export_to_srt(self, segments: List[AudioSegment], output_path: str) -> OperationResult:
        """SRT字幕形式でエクスポート"""
        try:
            def format_time(seconds):
                """SRT時間形式に変換"""
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
            
            with open(output_path, 'w', encoding='utf-8') as srtfile:
                for i, segment in enumerate(segments, 1):
                    speaker = segment.assigned_speaker or segment.top_speaker or 'unknown'
                    start_time = format_time(segment.start_time)
                    end_time = format_time(segment.end_time)
                    
                    srtfile.write(f"{i}\n")
                    srtfile.write(f"{start_time} --> {end_time}\n")
                    srtfile.write(f"{speaker}\n\n")
            
            return OperationResult(
                success=True,
                message=f"SRTファイルを保存しました: {output_path}"
            )
            
        except Exception as e:
            self.logger.exception("SRT保存エラー")
            return OperationResult(
                success=False,
                message=f"SRT保存エラー: {str(e)}",
                error_code=ErrorCode.EXPORT_FAILED
            )