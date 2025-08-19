"""
型定義モジュール
システム全体の型安全性を向上させる包括的な型定義
"""

from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, Dict, List, Tuple,
    Callable, Any, Awaitable, TypedDict, Literal, Final, 
    runtime_checkable, overload
)
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path


# === 基本型定義 ===

AudioData = np.ndarray
SampleRate = int
TimeStamp = float
Duration = float
Score = float
Confidence = float

FilePath = Union[str, Path]
SegmentID = int
SpeakerName = str


# === リテラル型 ===

AudioFormat = Literal['wav', 'mp3', 'flac', 'm4a', 'ogg']
DeviceType = Literal['auto', 'cuda', 'cpu', 'mps']
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
ConfidenceLevel = Literal['高', '中', '低']
TabName = Literal['single_speaker', 'multi_speaker', 'manual_segmentation', 'speaker_management', 'statistics']


# === TypedDict定義 ===

class RecognitionResultDict(TypedDict):
    """認識結果の型定義"""
    rank: int
    speaker: SpeakerName
    score: Score


class AudioSegmentDict(TypedDict):
    """音声セグメント辞書の型定義"""
    id: SegmentID
    start_time: TimeStamp
    end_time: TimeStamp
    duration: Duration
    assigned_speaker: Optional[SpeakerName]
    top_speaker: Optional[SpeakerName]
    top_score: Optional[Score]
    confidence_level: Optional[ConfidenceLevel]


class TimelineSegmentDict(TypedDict):
    """タイムライン表示用セグメント辞書"""
    start: TimeStamp
    end: TimeStamp
    duration: Duration
    confidence: Optional[Score]


class SpeakerStatsDict(TypedDict):
    """話者統計辞書"""
    segments: List[TimelineSegmentDict]
    total_time: Duration
    segment_count: int


class TimelineDataDict(TypedDict):
    """タイムラインデータ辞書"""
    speakers: Dict[SpeakerName, SpeakerStatsDict]
    total_duration: Duration
    segments: List[AudioSegmentDict]


class ConfigDict(TypedDict, total=False):
    """設定辞書（部分的な型定義）"""
    model: Dict[str, Any]
    audio: Dict[str, Any]
    recognition: Dict[str, Any]
    diarization: Dict[str, Any]
    ui: Dict[str, Any]


# === ジェネリック型変数 ===

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
R = TypeVar('R')

# 制約付き型変数
AudioType = TypeVar('AudioType', bound=np.ndarray)
ConfigType = TypeVar('ConfigType', bound='BaseConfig')
ResultType = TypeVar('ResultType', bound='BaseResult')


# === Protocol定義 ===

@runtime_checkable
class Recognizable(Protocol):
    """認識可能なオブジェクトのプロトコル"""
    
    def recognize(self, audio_data: AudioData, **kwargs) -> Any:
        """認識を実行"""
        ...


@runtime_checkable  
class Cacheable(Protocol[T]):
    """キャッシュ可能なオブジェクトのプロトコル"""
    
    def get_cache_key(self) -> str:
        """キャッシュキーを取得"""
        ...
    
    def serialize(self) -> bytes:
        """シリアライゼーション"""
        ...
    
    @classmethod
    def deserialize(cls, data: bytes) -> T:
        """デシリアライゼーション"""
        ...


@runtime_checkable
class Validatable(Protocol):
    """検証可能なオブジェクトのプロトコル"""
    
    def validate(self) -> List[str]:
        """検証してエラーメッセージのリストを返す"""
        ...
    
    def is_valid(self) -> bool:
        """有効性チェック"""
        ...


@runtime_checkable
class Exportable(Protocol):
    """エクスポート可能なオブジェクトのプロトコル"""
    
    def export_to_dict(self) -> Dict[str, Any]:
        """辞書形式でエクスポート"""
        ...
    
    def export_to_json(self) -> str:
        """JSON形式でエクスポート"""
        ...


@runtime_checkable
class AudioProcessor(Protocol):
    """音声処理のプロトコル"""
    
    def load_audio(self, file_path: FilePath) -> Tuple[AudioData, SampleRate]:
        """音声ファイル読み込み"""
        ...
    
    def save_audio(self, audio_data: AudioData, file_path: FilePath, sample_rate: SampleRate) -> None:
        """音声ファイル保存"""
        ...


@runtime_checkable  
class SegmentProcessor(Protocol):
    """セグメント処理のプロトコル"""
    
    def extract_segment(self, audio_data: AudioData, start_time: TimeStamp, 
                       end_time: TimeStamp, sample_rate: SampleRate) -> AudioData:
        """セグメント抽出"""
        ...


@runtime_checkable
class Recognizer(Protocol):
    """認識器のプロトコル"""
    
    def recognize_speaker(self, audio_path: FilePath) -> Any:
        """話者認識"""
        ...
    
    @property 
    def speaker_embeddings(self) -> Dict[SpeakerName, Any]:
        """話者エンベディング取得"""
        ...


@runtime_checkable
class AsyncRecognizer(Protocol):
    """非同期認識器のプロトコル"""
    
    async def recognize_speaker_async(self, audio_path: FilePath) -> Any:
        """非同期話者認識"""
        ...


@runtime_checkable
class ConfigurationProvider(Protocol[ConfigType]):
    """設定プロバイダーのプロトコル"""
    
    def get_config(self) -> ConfigType:
        """設定取得"""
        ...
    
    def update_config(self, **kwargs) -> bool:
        """設定更新"""
        ...
    
    def validate_config(self) -> List[str]:
        """設定検証"""
        ...


# === 抽象基底クラス ===

class BaseResult(ABC, Generic[T]):
    """結果基底クラス"""
    
    @abstractmethod
    def is_success(self) -> bool:
        """成功判定"""
        ...
    
    @abstractmethod
    def get_data(self) -> Optional[T]:
        """データ取得"""
        ...
    
    @abstractmethod
    def get_error_message(self) -> Optional[str]:
        """エラーメッセージ取得"""
        ...


class BaseConfig(ABC):
    """設定基底クラス"""
    
    @abstractmethod
    def validate(self) -> List[str]:
        """設定検証"""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        ...


class BaseManager(ABC, Generic[T]):
    """マネージャー基底クラス"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初期化"""
        ...
    
    @abstractmethod
    def cleanup(self) -> None:
        """クリーンアップ"""
        ...
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """状態取得"""
        ...


# === 関数型定義 ===

ProcessorFunc = Callable[[T], R]
AsyncProcessorFunc = Callable[[T], Awaitable[R]]
ValidatorFunc = Callable[[T], bool]
TransformerFunc = Callable[[T], T]
FilterFunc = Callable[[T], bool]

# 認識関連関数型
RecognitionFunc = Callable[[AudioData, SampleRate], Dict[SpeakerName, Score]]
AsyncRecognitionFunc = Callable[[AudioData, SampleRate], Awaitable[Dict[SpeakerName, Score]]]

# コールバック関数型
ErrorCallback = Callable[[Exception], None]
ProgressCallback = Callable[[float], None]  # 進捗率 0.0-1.0
CompletionCallback = Callable[[bool, Optional[str]], None]  # 成功フラグ、メッセージ


# === 列挙型 ===

class ProcessingStatus(Enum):
    """処理状態"""
    IDLE = "idle"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioQuality(Enum):
    """音声品質"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class CacheStrategy(Enum):
    """キャッシュ戦略"""
    LRU = "lru"
    TTL = "ttl"
    FIFO = "fifo"
    NO_CACHE = "no_cache"


# === データクラス ===

@dataclass(frozen=True)
class AudioMetadata:
    """音声メタデータ"""
    sample_rate: SampleRate
    duration: Duration
    channels: int
    format: AudioFormat
    file_size_mb: float
    quality: AudioQuality = AudioQuality.UNKNOWN


@dataclass(frozen=True)
class RecognitionMetrics:
    """認識メトリクス"""
    execution_time_ms: float
    confidence_score: Score
    top_candidates: int
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass(frozen=True)
class SegmentationResult:
    """セグメンテーション結果"""
    segment_id: SegmentID
    start_time: TimeStamp
    end_time: TimeStamp
    speaker: Optional[SpeakerName]
    confidence: Optional[Confidence]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingContext:
    """処理コンテキスト"""
    session_id: str
    user_id: Optional[str]
    timestamp: TimeStamp
    parameters: Dict[str, Any]
    metrics: Optional[RecognitionMetrics] = None


# === Union型 ===

AudioInput = Union[AudioData, FilePath, bytes]
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ProcessingResult = Union[SegmentationResult, Exception]
CacheValue = Union[AudioData, Dict[str, Any], List[Any]]


# === コンスタント ===

DEFAULT_SAMPLE_RATE: Final[SampleRate] = 16000
DEFAULT_CHUNK_SIZE: Final[int] = 1024
MAX_AUDIO_DURATION: Final[Duration] = 300.0  # 5分
MIN_SEGMENT_DURATION: Final[Duration] = 0.5   # 0.5秒
DEFAULT_CONFIDENCE_THRESHOLD: Final[Score] = 0.25


# === 型ガード関数 ===

def is_audio_data(obj: Any) -> bool:
    """AudioData型ガード"""
    return isinstance(obj, np.ndarray) and obj.ndim == 1


def is_valid_sample_rate(rate: Any) -> bool:
    """有効なサンプリングレート判定"""
    return isinstance(rate, int) and 8000 <= rate <= 96000


def is_valid_duration(duration: Any) -> bool:
    """有効な継続時間判定"""
    return isinstance(duration, (int, float)) and 0 < duration <= MAX_AUDIO_DURATION


def is_valid_confidence(confidence: Any) -> bool:
    """有効な信頼度判定"""
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0


# === ヘルパー関数 ===

def ensure_audio_format(format_str: str) -> AudioFormat:
    """音声フォーマットの保証"""
    format_lower = format_str.lower()
    if format_lower in ('wav', 'mp3', 'flac', 'm4a', 'ogg'):
        return format_lower  # type: ignore
    raise ValueError(f"サポートされていない音声フォーマット: {format_str}")


def ensure_file_path(path: Union[str, Path]) -> Path:
    """ファイルパスの保証"""
    return Path(path) if isinstance(path, str) else path


def ensure_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
    """正の数値の保証"""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name}は正の数値である必要があります: {value}")
    return value


# === デコレータ用型定義 ===

DecoratedFunc = TypeVar('DecoratedFunc', bound=Callable[..., Any])
AsyncDecoratedFunc = TypeVar('AsyncDecoratedFunc', bound=Callable[..., Awaitable[Any]])


# === 例外クラス ===

class TypeValidationError(TypeError):
    """型検証エラー"""
    pass


class ConfigurationError(ValueError):
    """設定エラー"""
    pass


class AudioProcessingError(RuntimeError):
    """音声処理エラー"""  
    pass