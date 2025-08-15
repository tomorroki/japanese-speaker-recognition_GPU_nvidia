"""
背景埋め込みローダー

JVSとCommon Voice日本語の事前計算済み埋め込みを読み込み、
AS-Norm等のスコア正規化に使用するモジュール
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import os

logger = logging.getLogger(__name__)

class BackgroundEmbeddingLoader:
    """背景埋め込みを管理するクラス"""
    
    def __init__(self, background_dir: str = "background_embeddings", config: Optional[dict] = None):
        """
        初期化
        
        Args:
            background_dir: 背景埋め込みファイルを格納するディレクトリ
            config: 設定辞書
        """
        self.background_dir = Path(background_dir)
        self.background_dir.mkdir(exist_ok=True)
        self.config = config or {}
        
        self.jvs_embeddings: Optional[np.ndarray] = None
        self.jvs_speaker_ids: Optional[np.ndarray] = None
        self.common_voice_embeddings: Optional[np.ndarray] = None
        
        logger.info(f"BackgroundEmbeddingLoader initialized with directory: {background_dir}")
    
    def load_jvs_embeddings(self, file_path: Optional[str] = None) -> bool:
        """
        JVS埋め込みを読み込み
        
        Args:
            file_path: 埋め込みファイルのパス（省略時は default を検索）
            
        Returns:
            読み込み成功フラグ
        """
        if file_path is None:
            # デフォルトファイルを検索
            candidates = [
                self.background_dir / "background_jvs_ecapa.npz",
                "background_jvs_ecapa.npz",
                self.background_dir / "jvs_embeddings.npz"
            ]
            
            file_path = None
            for candidate in candidates:
                if Path(candidate).exists():
                    file_path = candidate
                    break
        
        if file_path is None:
            logger.warning("JVS embeddings file not found")
            return False
        
        try:
            data = np.load(file_path)
            self.jvs_embeddings = data['embeddings']
            self.jvs_speaker_ids = data.get('speaker_ids', None)
            
            logger.info(f"Loaded JVS embeddings: {self.jvs_embeddings.shape} from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load JVS embeddings from {file_path}: {e}")
            return False
    
    def load_common_voice_embeddings(self, file_path: Optional[str] = None) -> bool:
        """
        Common Voice埋め込みを読み込み
        
        Args:
            file_path: 埋め込みファイルのパス（省略時は default を検索）
            
        Returns:
            読み込み成功フラグ
        """
        if file_path is None:
            # デフォルトファイルを検索
            candidates = [
                self.background_dir / "background_common_voice_ja_ecapa.npz",
                "background_common_voice_ja_ecapa.npz",
                self.background_dir / "common_voice_embeddings.npz"
            ]
            
            file_path = None
            for candidate in candidates:
                if Path(candidate).exists():
                    file_path = candidate
                    break
        
        if file_path is None:
            logger.warning("Common Voice embeddings file not found")
            return False
        
        try:
            data = np.load(file_path)
            self.common_voice_embeddings = data['embeddings']
            
            logger.info(f"Loaded Common Voice embeddings: {self.common_voice_embeddings.shape} from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Common Voice embeddings from {file_path}: {e}")
            return False
    
    def get_combined_background_embeddings(self, max_samples: Optional[int] = None) -> np.ndarray:
        """
        JVSとCommon Voiceの埋め込みを結合して返す
        
        Args:
            max_samples: 最大サンプル数（省略時は全て）
            
        Returns:
            結合された背景埋め込み配列
        """
        embeddings_list = []
        
        if self.jvs_embeddings is not None:
            embeddings_list.append(self.jvs_embeddings)
            logger.debug(f"Adding JVS embeddings: {self.jvs_embeddings.shape}")
        
        if self.common_voice_embeddings is not None:
            embeddings_list.append(self.common_voice_embeddings)
            logger.debug(f"Adding Common Voice embeddings: {self.common_voice_embeddings.shape}")
        
        if not embeddings_list:
            logger.warning("No background embeddings available")
            return np.array([])
        
        # 結合
        combined = np.vstack(embeddings_list)
        
        # サンプル数制限
        if max_samples is not None and len(combined) > max_samples:
            # ランダムサンプリング
            indices = np.random.choice(len(combined), max_samples, replace=False)
            combined = combined[indices]
            logger.info(f"Randomly sampled {max_samples} from {len(embeddings_list)} background embeddings")
        
        logger.info(f"Combined background embeddings shape: {combined.shape}")
        return combined
    
    def get_statistics(self) -> dict:
        """背景埋め込みの統計情報を取得"""
        stats = {
            'jvs_available': self.jvs_embeddings is not None,
            'common_voice_available': self.common_voice_embeddings is not None,
            'jvs_count': len(self.jvs_embeddings) if self.jvs_embeddings is not None else 0,
            'common_voice_count': len(self.common_voice_embeddings) if self.common_voice_embeddings is not None else 0,
            'total_count': 0,
            'embedding_dim': None
        }
        
        if self.jvs_embeddings is not None:
            stats['total_count'] += len(self.jvs_embeddings)
            stats['embedding_dim'] = self.jvs_embeddings.shape[1]
        
        if self.common_voice_embeddings is not None:
            stats['total_count'] += len(self.common_voice_embeddings)
            if stats['embedding_dim'] is None:
                stats['embedding_dim'] = self.common_voice_embeddings.shape[1]
        
        return stats
    
    def is_jvs_speaker(self, speaker_id: str) -> bool:
        """
        JVS話者かどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            JVS話者かどうか
        """
        # JVS話者パターン: jvs001, jvs002, ...
        return speaker_id.startswith('jvs') and len(speaker_id) >= 6 and speaker_id[3:].isdigit()
    
    def is_common_voice_speaker(self, speaker_id: str) -> bool:
        """
        Common Voice話者かどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            Common Voice話者かどうか
        """
        # Common Voice話者パターン
        cv_patterns = ['cv_', 'commonvoice_', 'mozilla_']
        return any(speaker_id.startswith(pattern) for pattern in cv_patterns)
    
    def should_exclude_speaker(self, speaker_id: str) -> bool:
        """
        話者を識別候補から除外すべきかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            除外すべきかどうか
        """
        # 設定確認
        datasets_config = self.config.get("datasets", {})
        exclude_background = datasets_config.get("exclude_background_speakers", True)
        
        if not exclude_background:
            return False
        
        # JVS話者の除外判定
        if self.is_jvs_speaker(speaker_id):
            allow_jvs = datasets_config.get("allow_jvs_speakers", False)
            if allow_jvs:
                return False
            return True
        
        # Common Voice話者の除外判定
        if self.is_common_voice_speaker(speaker_id):
            allow_cv = datasets_config.get("allow_common_voice_speakers", False)
            if allow_cv:
                return False
            return True
        
        return False
    
    def auto_load_all(self) -> Tuple[bool, bool]:
        """
        利用可能な背景埋め込みを自動で読み込み
        
        Returns:
            (JVS読み込み成功, Common Voice読み込み成功)
        """
        jvs_success = self.load_jvs_embeddings()
        cv_success = self.load_common_voice_embeddings()
        
        logger.info(f"Auto-load results: JVS={jvs_success}, Common Voice={cv_success}")
        return jvs_success, cv_success