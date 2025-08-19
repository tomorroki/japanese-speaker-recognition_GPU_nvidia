"""
パフォーマンス最適化モジュール
キャッシュ、並列処理、メモリ管理の最適化
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref
import gc
import psutil
import numpy as np

from logging_config import get_logger


T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_tasks: int = 0
    throughput_per_second: float = 0.0


class LRUCacheWithTTL(Generic[T]):
    """TTL付きLRUキャッシュ"""
    
    def __init__(self, maxsize: int = 128, ttl: float = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self.logger = get_logger(self.__class__.__name__)
    
    def get(self, key: str) -> Optional[T]:
        """キャッシュからアイテムを取得"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp <= self.ttl:
                    # ヒット - アイテムを最新に移動
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # 期限切れ
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: T) -> None:
        """キャッシュにアイテムを設定"""
        with self._lock:
            current_time = time.time()
            
            # 既存のキーを更新
            if key in self._cache:
                self._cache[key] = (value, current_time)
                self._cache.move_to_end(key)
            else:
                # 新しいキーを追加
                if len(self._cache) >= self.maxsize:
                    # 最古のアイテムを削除
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self.logger.debug(f"キャッシュからアイテムを削除: {oldest_key}")
                
                self._cache[key] = (value, current_time)
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl
        }


class AsyncBatchProcessor:
    """非同期バッチ処理器"""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_workers: int = 4,
                 timeout: float = 300.0):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
    
    async def process_batch(self, 
                           items: List[Any],
                           processor_func: Callable,
                           *args, **kwargs) -> List[Any]:
        """バッチを並列処理"""
        if not items:
            return []
        
        results = []
        total_batches = len(items) // self.batch_size + (1 if len(items) % self.batch_size else 0)
        
        self.logger.info(f"バッチ処理開始: {len(items)}アイテム, {total_batches}バッチ")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # バッチに分割
            batches = [
                items[i:i + self.batch_size] 
                for i in range(0, len(items), self.batch_size)
            ]
            
            # 並列実行
            tasks = []
            for batch in batches:
                task = asyncio.get_event_loop().run_in_executor(
                    executor, self._process_single_batch, batch, processor_func, args, kwargs
                )
                tasks.append(task)
            
            # 結果収集
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.timeout
                )
                
                # 結果をフラット化
                for batch_result in batch_results:
                    if isinstance(batch_result, Exception):
                        self.logger.error(f"バッチ処理エラー: {str(batch_result)}")
                        continue
                    results.extend(batch_result)
                
            except asyncio.TimeoutError:
                self.logger.error(f"バッチ処理がタイムアウトしました: {self.timeout}秒")
                raise
        
        self.logger.info(f"バッチ処理完了: {len(results)}件の結果")
        return results
    
    def _process_single_batch(self, batch: List[Any], processor_func: Callable, 
                             args: tuple, kwargs: dict) -> List[Any]:
        """単一バッチの処理"""
        results = []
        for item in batch:
            try:
                result = processor_func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"アイテム処理エラー: {str(e)}")
                results.append(None)  # エラー時はNoneを返す
        return results


class MemoryManager:
    """メモリ管理器"""
    
    def __init__(self, memory_limit_mb: int = 2048, cleanup_threshold: float = 0.8):
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.logger = get_logger(self.__class__.__name__)
        self._weak_refs: List[weakref.ref] = []
    
    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """メモリ制限をチェック"""
        current_usage = self.get_memory_usage()
        limit_threshold = self.memory_limit_mb * self.cleanup_threshold
        
        if current_usage > limit_threshold:
            self.logger.warning(f"メモリ使用量が閾値を超過: {current_usage:.1f}MB > {limit_threshold:.1f}MB")
            return False
        return True
    
    def cleanup_memory(self, force_gc: bool = True) -> float:
        """メモリクリーンアップ"""
        before_usage = self.get_memory_usage()
        
        # 弱参照のクリーンアップ
        self._cleanup_weak_refs()
        
        # ガベージコレクション
        if force_gc:
            collected = gc.collect()
            self.logger.debug(f"ガベージコレクションで{collected}個のオブジェクトを回収")
        
        after_usage = self.get_memory_usage()
        freed_mb = before_usage - after_usage
        
        self.logger.info(f"メモリクリーンアップ完了: {freed_mb:.1f}MB解放")
        return freed_mb
    
    def register_for_cleanup(self, obj: Any) -> None:
        """オブジェクトをクリーンアップ対象として登録"""
        self._weak_refs.append(weakref.ref(obj))
    
    def _cleanup_weak_refs(self) -> None:
        """無効な弱参照をクリーンアップ"""
        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
    
    @asynccontextmanager
    async def memory_monitor(self):
        """メモリ監視コンテキストマネージャ"""
        initial_usage = self.get_memory_usage()
        self.logger.debug(f"メモリ監視開始: {initial_usage:.1f}MB")
        
        try:
            yield
        finally:
            final_usage = self.get_memory_usage()
            usage_diff = final_usage - initial_usage
            self.logger.debug(f"メモリ監視終了: {final_usage:.1f}MB (差分: {usage_diff:+.1f}MB)")
            
            if not self.check_memory_limit():
                self.cleanup_memory()


class PerformanceProfiler:
    """パフォーマンスプロファイラ"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def profile_function(self, func_name: str):
        """関数プロファイリングデコレータ"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # メトリクス計算
                    execution_time = time.time() - start_time
                    memory_usage = self._get_memory_usage() - start_memory
                    cpu_usage = self._get_cpu_usage() - start_cpu
                    
                    # メトリクス保存
                    if func_name not in self.metrics:
                        self.metrics[func_name] = PerformanceMetrics()
                    
                    metrics = self.metrics[func_name]
                    metrics.execution_time = execution_time
                    metrics.memory_usage_mb = memory_usage
                    metrics.cpu_usage_percent = cpu_usage
                    
                    self.logger.debug(f"プロファイル結果 {func_name}: "
                                    f"実行時間={execution_time:.3f}s, "
                                    f"メモリ={memory_usage:.1f}MB, "
                                    f"CPU={cpu_usage:.1f}%")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"プロファイル中にエラー {func_name}: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """メモリ使用量を取得"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """CPU使用率を取得"""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def get_metrics_report(self) -> Dict[str, Dict[str, Any]]:
        """メトリクスレポートを取得"""
        report = {}
        for func_name, metrics in self.metrics.items():
            report[func_name] = {
                "execution_time": metrics.execution_time,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "cache_hit_rate": metrics.cache_hit_rate,
                "throughput_per_second": metrics.throughput_per_second
            }
        return report
    
    def reset_metrics(self) -> None:
        """メトリクスをリセット"""
        self.metrics.clear()
        self.logger.info("パフォーマンスメトリクスをリセットしました")


# グローバルインスタンス
_memory_manager = MemoryManager()
_profiler = PerformanceProfiler()
_batch_processor = AsyncBatchProcessor()


def get_memory_manager() -> MemoryManager:
    """グローバルメモリマネージャーを取得"""
    return _memory_manager


def get_profiler() -> PerformanceProfiler:
    """グローバルプロファイラを取得"""
    return _profiler


def get_batch_processor() -> AsyncBatchProcessor:
    """グローバルバッチプロセッサを取得"""
    return _batch_processor


def profile_performance(func_name: str):
    """パフォーマンスプロファイリングデコレータ"""
    return _profiler.profile_function(func_name)


def memory_optimized(cleanup_threshold: float = 0.8):
    """メモリ最適化デコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with _memory_manager.memory_monitor():
                if not _memory_manager.check_memory_limit():
                    _memory_manager.cleanup_memory()
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _memory_manager.check_memory_limit():
                _memory_manager.cleanup_memory()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# キャッシュファクトリー
def create_cache(maxsize: int = 128, ttl: float = 3600) -> LRUCacheWithTTL:
    """TTL付きキャッシュを作成"""
    return LRUCacheWithTTL(maxsize=maxsize, ttl=ttl)