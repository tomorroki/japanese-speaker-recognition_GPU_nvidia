"""
ロギング設定モジュール
システム全体のログ設定を一元管理
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

class LoggingConfig:
    """ログ設定クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "logging_config.json"
        self.log_dir = Path("logs")
        self.default_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "%(levelname)s - %(name)s - %(message)s"
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/application.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8"
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler", 
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8"
                },
                "segmentation_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed", 
                    "filename": "logs/segmentation.log",
                    "maxBytes": 5242880,  # 5MB
                    "backupCount": 3,
                    "encoding": "utf8"
                }
            },
            "loggers": {
                "ManualSpeakerSegmentatorV2": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "segmentation_file"],
                    "propagate": False
                },
                "RecognitionManager": {
                    "level": "DEBUG", 
                    "handlers": ["console", "file", "segmentation_file"],
                    "propagate": False
                },
                "AudioLoader": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "SegmentManager": {
                    "level": "INFO",
                    "handlers": ["console", "file", "segmentation_file"], 
                    "propagate": False
                },
                "ExportManager": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console", "error_file"]
            }
        }
    
    def setup_logging(self, config: Optional[Dict[str, Any]] = None) -> None:
        """ログ設定の初期化"""
        try:
            # ログディレクトリの作成
            self.log_dir.mkdir(exist_ok=True)
            
            # 設定ファイルの読み込みまたはデフォルト使用
            if config:
                log_config = config
            elif Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    log_config = json.load(f)
            else:
                log_config = self.default_config
                self.save_default_config()
            
            # ファイルパスの絶対パス変換
            for handler_name, handler_config in log_config.get("handlers", {}).items():
                if "filename" in handler_config:
                    filename = handler_config["filename"]
                    if not os.path.isabs(filename):
                        handler_config["filename"] = str(Path(filename).resolve())
            
            # ログ設定適用
            logging.config.dictConfig(log_config)
            
            # 初期化成功をログ出力
            logger = logging.getLogger(__name__)
            logger.info("ログシステム初期化完了")
            
        except Exception as e:
            # ログ設定に失敗した場合はフォールバック設定
            self.setup_fallback_logging()
            logger = logging.getLogger(__name__)
            logger.error(f"ログ設定に失敗、フォールバック設定を使用: {str(e)}")
    
    def setup_fallback_logging(self) -> None:
        """フォールバックログ設定"""
        # 基本的なコンソール出力のみ
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def save_default_config(self) -> None:
        """デフォルト設定をファイルに保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"デフォルト設定の保存に失敗: {str(e)}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """ロガーインスタンスの取得"""
        return logging.getLogger(name)
    
    def set_log_level(self, logger_name: str, level: str) -> bool:
        """動的ログレベル変更"""
        try:
            logger = logging.getLogger(logger_name)
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"無効なログレベル: {level}")
            
            logger.setLevel(numeric_level)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ログレベル変更に失敗: {str(e)}")
            return False
    
    def add_stream_handler(self, logger_name: str, stream=None, 
                          level: str = "INFO", format_name: str = "simple") -> bool:
        """動的ストリームハンドラー追加"""
        try:
            logger = logging.getLogger(logger_name)
            
            handler = logging.StreamHandler(stream or sys.stdout)
            handler.setLevel(getattr(logging, level.upper()))
            
            # フォーマッター設定
            if format_name == "detailed":
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
            else:
                formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            return True
        except Exception as e:
            print(f"ストリームハンドラー追加に失敗: {str(e)}")
            return False
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """古いログファイルのクリーンアップ"""
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    
            logger = logging.getLogger(__name__)
            logger.info(f"{days}日以上古いログファイルを削除しました")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ログファイルクリーンアップに失敗: {str(e)}")


# グローバルログ設定インスタンス
_logging_config = LoggingConfig()


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """グローバルログ設定の初期化"""
    _logging_config.setup_logging(config)


def get_logger(name: str) -> logging.Logger:
    """ロガーインスタンスの取得"""
    return _logging_config.get_logger(name)


def set_log_level(logger_name: str, level: str) -> bool:
    """ログレベルの動的変更"""
    return _logging_config.set_log_level(logger_name, level)


def cleanup_old_logs(days: int = 30) -> None:
    """古いログファイルのクリーンアップ"""
    _logging_config.cleanup_old_logs(days)


# コンテキストマネージャーでのロギング
class LoggingContext:
    """ロギングコンテキストマネージャー"""
    
    def __init__(self, logger_name: str, operation_name: str, 
                 log_entry: bool = True, log_exit: bool = True, 
                 log_errors: bool = True):
        self.logger = get_logger(logger_name)
        self.operation_name = operation_name
        self.log_entry = log_entry
        self.log_exit = log_exit 
        self.log_errors = log_errors
        self.start_time = None
    
    def __enter__(self):
        if self.log_entry:
            self.start_time = datetime.now()
            self.logger.info(f"{self.operation_name}開始")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.log_errors:
            self.logger.error(f"{self.operation_name}エラー: {str(exc_val)}", exc_info=True)
        elif self.log_exit:
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.logger.info(f"{self.operation_name}完了 (実行時間: {duration:.2f}秒)")
            else:
                self.logger.info(f"{self.operation_name}完了")
        
        return False  # 例外は再発生させる