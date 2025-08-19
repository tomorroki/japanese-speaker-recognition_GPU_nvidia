"""
設定管理モジュール
アプリケーション全体の設定を一元管理
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, fields
import logging
from copy import deepcopy

# from logging_config import get_logger  # 循環依存を避けるため一時的にコメントアウト


@dataclass
class ModelConfig:
    """モデル関連設定"""
    speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    device: str = "auto"  # auto, cuda, cpu, mps
    cache_dir: Optional[str] = None
    model_revision: str = "main"


@dataclass
class AudioConfig:
    """音声処理設定"""
    sample_rate: int = 16000
    duration_min: float = 2.0
    duration_max: float = 30.0
    normalize: bool = True
    mono: bool = True
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['wav', 'mp3', 'flac', 'm4a', 'ogg']


@dataclass
class RecognitionConfig:
    """認識処理設定"""
    threshold: float = 0.25
    use_score_normalization: bool = True
    background_speakers_count: int = 100
    max_recognition_results: int = 5
    confidence_thresholds: dict = None
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                'high': 0.7,
                'medium': 0.4,
                'low': 0.0
            }


@dataclass
class DiarizationConfig:
    """話者分離設定"""
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 10
    min_segment_duration: float = 0.5
    use_auth_token: bool = True


@dataclass
class SegmentationConfig:
    """セグメンテーション設定"""
    min_segment_duration: float = 0.5
    max_segment_duration: float = 300.0
    allow_overlap: bool = False
    auto_merge_threshold: float = 0.1
    validation_enabled: bool = True


@dataclass
class ProcessingConfig:
    """処理関連設定"""
    target_sample_rate: int = 16000
    normalize: bool = True
    max_parallel_workers: int = 4
    chunk_size: int = 1024
    use_gpu_acceleration: bool = True
    memory_limit_mb: int = 2048


@dataclass
class DatasetConfig:
    """データセット設定"""
    exclude_background_speakers: bool = True
    allow_jvs_speakers: bool = True
    allow_common_voice_speakers: bool = False
    jvs_path: Optional[str] = None
    common_voice_path: Optional[str] = None
    enrollment_path: str = "enroll"
    background_embeddings_path: str = "."


@dataclass
class UIConfig:
    """UI設定"""
    show_jvs_in_results: bool = False
    show_common_voice_in_results: bool = False
    default_tab: str = "single_speaker"
    theme: str = "auto"
    language: str = "ja"
    page_title: str = "日本語話者認識システム"
    page_icon: str = "🎤"
    layout: str = "wide"


@dataclass
class CacheConfig:
    """キャッシュ設定"""
    enable_cache: bool = True
    max_cache_size: int = 100
    cache_dir: str = "cache"
    speaker_embeddings_cache: str = "enrolled_speakers_embeddings.npz"
    background_embeddings_cache: str = "background_embeddings"
    audio_cache_duration: int = 3600  # seconds


@dataclass
class LoggingConfigData:
    """ログ設定"""
    level: str = "INFO"
    file_enabled: bool = True
    console_enabled: bool = True
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    format_detailed: bool = True


@dataclass
class SecurityConfig:
    """セキュリティ設定"""
    max_file_size_mb: int = 100
    allowed_file_extensions: list = None
    upload_timeout: int = 300
    rate_limit_enabled: bool = False
    max_requests_per_hour: int = 100
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']


@dataclass
class ApplicationConfig:
    """アプリケーション全体設定"""
    model: ModelConfig = None
    audio: AudioConfig = None
    recognition: RecognitionConfig = None
    diarization: DiarizationConfig = None
    segmentation: SegmentationConfig = None
    processing: ProcessingConfig = None
    datasets: DatasetConfig = None
    ui: UIConfig = None
    cache: CacheConfig = None
    logging: LoggingConfigData = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.recognition is None:
            self.recognition = RecognitionConfig()
        if self.diarization is None:
            self.diarization = DiarizationConfig()
        if self.segmentation is None:
            self.segmentation = SegmentationConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.datasets is None:
            self.datasets = DatasetConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.logging is None:
            self.logging = LoggingConfigData()
        if self.security is None:
            self.security = SecurityConfig()


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_file: str = "config.json", env_prefix: str = "SPEAKER_RECOG_"):
        self.config_file = Path(config_file)
        self.env_prefix = env_prefix
        self.logger = logging.getLogger(__name__)
        self._config: Optional[ApplicationConfig] = None
        self._file_watcher_enabled = False
        
    def load_config(self, create_if_missing: bool = True) -> ApplicationConfig:
        """設定の読み込み"""
        try:
            if self.config_file.exists():
                self.logger.info(f"設定ファイル読み込み: {self.config_file}")
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                self._config = self._dict_to_config(config_dict)
                
            elif create_if_missing:
                self.logger.info("設定ファイルが見つからないため、デフォルト設定を作成")
                self._config = ApplicationConfig()
                self.save_config()
            else:
                self._config = ApplicationConfig()
            
            # 環境変数による設定上書き
            self._apply_env_overrides()
            
            self.logger.info("設定読み込み完了")
            return self._config
            
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {str(e)}")
            self._config = ApplicationConfig()
            return self._config
    
    def save_config(self, config: Optional[ApplicationConfig] = None) -> bool:
        """設定の保存"""
        try:
            target_config = config or self._config
            if target_config is None:
                self.logger.error("保存する設定がありません")
                return False
            
            # バックアップ作成
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.backup')
                self.config_file.replace(backup_file)
                self.logger.debug(f"設定ファイルバックアップ作成: {backup_file}")
            
            config_dict = self._config_to_dict(target_config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定保存完了: {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定保存エラー: {str(e)}")
            return False
    
    def get_config(self) -> ApplicationConfig:
        """現在の設定を取得"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """設定値の更新"""
        try:
            if self._config is None:
                self.load_config()
            
            if hasattr(self._config, section):
                section_obj = getattr(self._config, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    self.logger.info(f"設定更新: {section}.{key} = {value}")
                    return True
                else:
                    self.logger.warning(f"不明な設定キー: {section}.{key}")
                    return False
            else:
                self.logger.warning(f"不明な設定セクション: {section}")
                return False
                
        except Exception as e:
            self.logger.error(f"設定更新エラー: {str(e)}")
            return False
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """設定値の取得"""
        try:
            if self._config is None:
                self.load_config()
            
            if hasattr(self._config, section):
                section_obj = getattr(self._config, section)
                return getattr(section_obj, key, default)
            
            return default
            
        except Exception as e:
            self.logger.error(f"設定取得エラー: {str(e)}")
            return default
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ApplicationConfig:
        """辞書から設定オブジェクトへの変換"""
        config = ApplicationConfig()
        
        # 各セクションの設定
        section_classes = {
            'model': ModelConfig,
            'audio': AudioConfig,
            'recognition': RecognitionConfig,
            'diarization': DiarizationConfig,
            'segmentation': SegmentationConfig,
            'processing': ProcessingConfig,
            'datasets': DatasetConfig,
            'ui': UIConfig,
            'cache': CacheConfig,
            'logging': LoggingConfigData,
            'security': SecurityConfig
        }
        
        for section_name, section_class in section_classes.items():
            if section_name in config_dict:
                section_data = config_dict[section_name]
                if isinstance(section_data, dict):
                    # dataclassのフィールドのみを抽出
                    field_names = {f.name for f in fields(section_class)}
                    filtered_data = {k: v for k, v in section_data.items() if k in field_names}
                    section_obj = section_class(**filtered_data)
                    setattr(config, section_name, section_obj)
        
        return config
    
    def _config_to_dict(self, config: ApplicationConfig) -> Dict[str, Any]:
        """設定オブジェクトから辞書への変換"""
        return asdict(config)
    
    def _apply_env_overrides(self) -> None:
        """環境変数による設定上書き"""
        if self._config is None:
            return
        
        # 環境変数のマッピング
        env_mappings = {
            f"{self.env_prefix}MODEL_DEVICE": ("model", "device"),
            f"{self.env_prefix}AUDIO_SAMPLE_RATE": ("audio", "sample_rate"),
            f"{self.env_prefix}RECOGNITION_THRESHOLD": ("recognition", "threshold"),
            f"{self.env_prefix}PROCESSING_MAX_WORKERS": ("processing", "max_parallel_workers"),
            f"{self.env_prefix}UI_SHOW_JVS": ("ui", "show_jvs_in_results"),
            f"{self.env_prefix}LOG_LEVEL": ("logging", "level"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 型変換
                try:
                    section_obj = getattr(self._config, section)
                    current_value = getattr(section_obj, key)
                    
                    if isinstance(current_value, bool):
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        converted_value = int(env_value)
                    elif isinstance(current_value, float):
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    setattr(section_obj, key, converted_value)
                    self.logger.info(f"環境変数による設定上書き: {section}.{key} = {converted_value}")
                    
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"環境変数変換エラー {env_var}: {str(e)}")
    
    def validate_config(self) -> List[str]:
        """設定の検証"""
        errors = []
        
        if self._config is None:
            errors.append("設定が読み込まれていません")
            return errors
        
        # 各種検証ルール
        if self._config.audio.sample_rate <= 0:
            errors.append("audio.sample_rate は正の値である必要があります")
        
        if self._config.audio.duration_min >= self._config.audio.duration_max:
            errors.append("audio.duration_min は duration_max より小さい必要があります")
        
        if not (0.0 <= self._config.recognition.threshold <= 1.0):
            errors.append("recognition.threshold は 0.0 から 1.0 の間である必要があります")
        
        if self._config.processing.max_parallel_workers <= 0:
            errors.append("processing.max_parallel_workers は正の値である必要があります")
        
        if self._config.security.max_file_size_mb <= 0:
            errors.append("security.max_file_size_mb は正の値である必要があります")
        
        # パス存在確認
        if self._config.datasets.enrollment_path:
            enrollment_path = Path(self._config.datasets.enrollment_path)
            if not enrollment_path.exists():
                errors.append(f"enrollment_path が見つかりません: {enrollment_path}")
        
        return errors
    
    def reset_to_defaults(self) -> ApplicationConfig:
        """設定をデフォルトにリセット"""
        self._config = ApplicationConfig()
        self.logger.info("設定をデフォルトにリセットしました")
        return self._config
    
    def export_config(self, export_path: Union[str, Path]) -> bool:
        """設定のエクスポート"""
        try:
            export_file = Path(export_path)
            if self._config is None:
                self.load_config()
            
            config_dict = self._config_to_dict(self._config)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定エクスポート完了: {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定エクスポートエラー: {str(e)}")
            return False


# グローバル設定マネージャー
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """グローバル設定マネージャーの取得"""
    return _config_manager


def get_config() -> ApplicationConfig:
    """現在の設定を取得"""
    return _config_manager.get_config()


def update_config(section: str, key: str, value: Any) -> bool:
    """設定値の更新"""
    return _config_manager.update_config(section, key, value)


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """設定値の取得"""
    return _config_manager.get_value(section, key, default)