"""
セグメント切り出し・前処理専用モジュール
"""

import librosa
import numpy as np
import soundfile as sf
import logging


class SegmentProcessor:
    """音声セグメント処理クラス"""
    
    def __init__(self, target_sr: int = 16000):
        """
        初期化
        
        Args:
            target_sr: ターゲットサンプリングレート
        """
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
    
    def extract_segment(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """
        時間帯セグメント切り出し・前処理
        
        Args:
            audio_path: 元音声ファイルパス
            start: 開始時間（秒）
            end: 終了時間（秒）
            
        Returns:
            前処理済み音声セグメント（16kHz, 正規化済み）
        """
        try:
            # 1. 元音声読み込み（元のサンプリングレートで）
            audio, original_sr = librosa.load(audio_path, sr=None)
            
            # 2. 時間帯切り出し
            start_sample = int(start * original_sr)
            end_sample = int(end * original_sr)
            
            # 範囲チェック
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if start_sample >= end_sample:
                self.logger.warning(f"Invalid segment range: {start}s-{end}s")
                return np.array([])
            
            segment = audio[start_sample:end_sample]
            
            # 3. 話者認識用前処理
            # リサンプリング
            if original_sr != self.target_sr:
                segment = librosa.resample(
                    segment, 
                    orig_sr=original_sr, 
                    target_sr=self.target_sr
                )
            
            # 正規化
            if len(segment) > 0:
                segment = librosa.util.normalize(segment)
            
            return segment
            
        except Exception as e:
            self.logger.error(f"セグメント抽出エラー: {e}")
            return np.array([])
    
    def save_segment(self, segment: np.ndarray, output_path: str) -> bool:
        """
        セグメント保存
        
        Args:
            segment: 音声セグメント
            output_path: 出力パス
            
        Returns:
            保存成功の可否
        """
        try:
            sf.write(output_path, segment, self.target_sr)
            return True
        except Exception as e:
            self.logger.error(f"セグメント保存エラー: {e}")
            return False
    
    def validate_segment(self, segment: np.ndarray, min_duration: float = 0.5) -> bool:
        """
        セグメントの有効性チェック
        
        Args:
            segment: 音声セグメント
            min_duration: 最小時間長（秒）
            
        Returns:
            有効性
        """
        if len(segment) == 0:
            return False
        
        duration = len(segment) / self.target_sr
        return duration >= min_duration