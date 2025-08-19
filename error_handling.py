"""
エラーハンドリング強化モジュール
システム全体の例外処理とエラーレポート機能
"""

import logging
import traceback
import sys
from typing import Any, Optional, Dict, Callable, Tuple, Type
from functools import wraps
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

from logging_config import get_logger


class ErrorSeverity(Enum):
    """エラー重要度レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    AUDIO_PROCESSING = "audio_processing"
    RECOGNITION = "recognition"
    SEGMENTATION = "segmentation"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorReport:
    """エラーレポート構造"""
    timestamp: str
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    function_name: str
    file_name: str
    line_number: int
    traceback_str: str
    context: Dict[str, Any]
    user_message: str
    recovery_suggestion: str


class ErrorHandler:
    """エラーハンドリングクラス"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.error_reports_dir = Path("error_reports")
        self.error_reports_dir.mkdir(exist_ok=True)
        self.error_callbacks: Dict[ErrorCategory, Callable] = {}
        
        # エラーメッセージのマッピング
        self.error_messages = {
            "AUDIO_LOAD_FAILED": {
                "user": "音声ファイルの読み込みに失敗しました。ファイル形式を確認してください。",
                "recovery": "サポートされている形式（WAV, MP3, FLAC, M4A, OGG）を使用してください。"
            },
            "RECOGNITION_FAILED": {
                "user": "話者認識に失敗しました。音声品質を確認してください。", 
                "recovery": "ノイズの少ない音声を使用するか、音量を調整してください。"
            },
            "SEGMENT_OVERLAP": {
                "user": "セグメントの時間が重複しています。",
                "recovery": "既存のセグメントと重複しない時間範囲を選択してください。"
            },
            "INVALID_TIME_RANGE": {
                "user": "無効な時間範囲が指定されました。",
                "recovery": "開始時間は終了時間より前で、音声の範囲内に収まるようにしてください。"
            },
            "EXPORT_FAILED": {
                "user": "エクスポートに失敗しました。",
                "recovery": "書き込み権限のあるフォルダを選択し、十分な空き容量があることを確認してください。"
            }
        }
    
    def handle_error(self, 
                    error: Exception,
                    category: ErrorCategory,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    custom_message: Optional[str] = None) -> ErrorReport:
        """エラーの包括的な処理"""
        
        # トレースバック情報取得
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback:
            tb_frame = exc_traceback.tb_frame
            function_name = tb_frame.f_code.co_name
            file_name = tb_frame.f_code.co_filename
            line_number = exc_traceback.tb_lineno
        else:
            function_name = "unknown"
            file_name = "unknown"
            line_number = 0
        
        traceback_str = traceback.format_exc()
        
        # エラーレポート生成
        error_report = ErrorReport(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            function_name=function_name,
            file_name=file_name,
            line_number=line_number,
            traceback_str=traceback_str,
            context=context or {},
            user_message=custom_message or self._get_user_message(str(error)),
            recovery_suggestion=self._get_recovery_suggestion(str(error))
        )
        
        # ログ出力
        self._log_error(error_report)
        
        # エラーレポート保存
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._save_error_report(error_report)
        
        # カテゴリ別コールバック実行
        if category in self.error_callbacks:
            try:
                self.error_callbacks[category](error_report)
            except Exception as callback_error:
                self.logger.error(f"エラーコールバック実行失敗: {str(callback_error)}")
        
        return error_report
    
    def _get_user_message(self, error_message: str) -> str:
        """ユーザー向けエラーメッセージの生成"""
        for error_code, messages in self.error_messages.items():
            if error_code.lower() in error_message.lower():
                return messages["user"]
        return "予期しないエラーが発生しました。"
    
    def _get_recovery_suggestion(self, error_message: str) -> str:
        """復旧提案の生成"""
        for error_code, messages in self.error_messages.items():
            if error_code.lower() in error_message.lower():
                return messages["recovery"]
        return "システム管理者にお問い合わせください。"
    
    def _log_error(self, error_report: ErrorReport) -> None:
        """エラーのログ出力"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_report.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"[{error_report.category.value}] {error_report.error_type}: {error_report.error_message}",
            extra={
                "function_name": error_report.function_name,
                "file_name": error_report.file_name,
                "line_number": error_report.line_number,
                "context": error_report.context
            }
        )
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR - 即座の対応が必要: {error_report.traceback_str}")
    
    def _save_error_report(self, error_report: ErrorReport) -> None:
        """エラーレポートのファイル保存"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.error_reports_dir / f"error_report_{timestamp_str}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(error_report), f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"エラーレポート保存完了: {report_file}")
            
        except Exception as save_error:
            self.logger.error(f"エラーレポート保存失敗: {str(save_error)}")
    
    def register_error_callback(self, category: ErrorCategory, callback: Callable) -> None:
        """エラーカテゴリ別コールバック登録"""
        self.error_callbacks[category] = callback
        self.logger.info(f"エラーコールバック登録: {category.value}")
    
    def add_custom_error_message(self, error_code: str, user_message: str, recovery_suggestion: str) -> None:
        """カスタムエラーメッセージの追加"""
        self.error_messages[error_code] = {
            "user": user_message,
            "recovery": recovery_suggestion
        }


# グローバルエラーハンドラーインスタンス
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """グローバルエラーハンドラーの取得"""
    return _error_handler


def handle_error(error: Exception,
                category: ErrorCategory,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[Dict[str, Any]] = None,
                custom_message: Optional[str] = None) -> ErrorReport:
    """エラー処理の簡易関数"""
    return _error_handler.handle_error(error, category, severity, context, custom_message)


def safe_execute(func: Callable,
                category: ErrorCategory,
                *args,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[Dict[str, Any]] = None,
                default_return: Any = None,
                **kwargs) -> Tuple[bool, Any, Optional[ErrorReport]]:
    """安全な関数実行"""
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        error_report = handle_error(e, category, severity, context)
        return False, default_return, error_report


def error_handler_decorator(category: ErrorCategory,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           return_on_error: Any = None,
                           log_entry: bool = False):
    """エラーハンドリングデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            if log_entry:
                logger.info(f"{func.__name__}実行開始")
            
            try:
                result = func(*args, **kwargs)
                if log_entry:
                    logger.info(f"{func.__name__}実行完了")
                return result
                
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # 長すぎる場合は切り捨て
                    "kwargs": str(kwargs)[:200]
                }
                
                error_report = handle_error(e, category, severity, context)
                logger.error(f"{func.__name__}でエラー発生: {error_report.user_message}")
                
                if return_on_error is not None:
                    return return_on_error
                else:
                    # エラーを再発生
                    raise
        
        return wrapper
    return decorator


def retry_with_backoff(max_attempts: int = 3,
                      backoff_factor: float = 1.0,
                      category: ErrorCategory = ErrorCategory.SYSTEM):
    """指数バックオフ付きリトライデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            logger = get_logger(func.__module__)
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # 最後の試行で失敗した場合はエラー処理
                        context = {
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts
                        }
                        handle_error(e, category, ErrorSeverity.HIGH, context)
                        raise
                    else:
                        # リトライ前の待機
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"{func.__name__} 試行{attempt + 1}失敗、{wait_time}秒後にリトライ: {str(e)}")
                        time.sleep(wait_time)
            
        return wrapper
    return decorator


class ValidationError(Exception):
    """バリデーションエラー"""
    pass


class ConfigurationError(Exception):
    """設定エラー"""
    pass


class AudioProcessingError(Exception):
    """音声処理エラー"""
    pass


class RecognitionError(Exception):
    """認識処理エラー"""
    pass


class SegmentationError(Exception):
    """セグメンテーションエラー"""
    pass