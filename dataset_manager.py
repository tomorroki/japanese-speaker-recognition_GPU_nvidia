"""
データセット管理モジュール
JVSとCommon Voice日本語データセットの話者ID管理
"""

import os
import json
import logging
from typing import List, Set, Dict, Any
from pathlib import Path

class DatasetManager:
    """データセット管理クラス"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # JVS話者IDリスト
        self.jvs_speaker_ids = set(self.config["datasets"]["jvs_speaker_ids"])
        
        # Common Voice話者IDパターン
        self.cv_patterns = self.config["datasets"]["common_voice_patterns"]
        
        # 背景話者除外設定
        self.exclude_background = self.config["datasets"]["exclude_background_speakers"]
        
        self.logger.info(f"DatasetManager initialized with {len(self.jvs_speaker_ids)} JVS speakers")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"設定ファイルの形式が不正です: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config["logging"]["format"])
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def is_jvs_speaker(self, speaker_id: str) -> bool:
        """
        JVS話者かどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            JVS話者の場合True
        """
        return speaker_id.lower() in self.jvs_speaker_ids
    
    def is_common_voice_speaker(self, speaker_id: str) -> bool:
        """
        Common Voice話者かどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            Common Voice話者の場合True
        """
        speaker_id_lower = speaker_id.lower()
        return any(pattern in speaker_id_lower for pattern in self.cv_patterns)
    
    def is_background_speaker(self, speaker_id: str) -> bool:
        """
        背景話者（除外対象）かどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            背景話者の場合True
        """
        return (self.is_jvs_speaker(speaker_id) or 
                self.is_common_voice_speaker(speaker_id))
    
    def should_exclude_speaker(self, speaker_id: str) -> bool:
        """
        話者を除外すべきかどうかを判定
        
        Args:
            speaker_id: 話者ID
            
        Returns:
            除外すべき場合True
        """
        if not self.exclude_background:
            return False
        
        # JVS話者の場合
        if self.is_jvs_speaker(speaker_id):
            # allow_jvs_speakersがTrueなら除外しない
            return not self.config["datasets"]["allow_jvs_speakers"]
        
        # Common Voice話者の場合
        if self.is_common_voice_speaker(speaker_id):
            # allow_common_voice_speakersがTrueなら除外しない
            return not self.config["datasets"]["allow_common_voice_speakers"]
        
        # その他の話者は除外しない
        return False
    
    def get_valid_speakers(self, enroll_dir: str) -> List[str]:
        """
        登録可能な話者IDリストを取得
        
        Args:
            enroll_dir: 登録用ディレクトリパス
            
        Returns:
            有効な話者IDのリスト
        """
        valid_speakers = []
        
        if not os.path.exists(enroll_dir):
            self.logger.warning(f"登録ディレクトリが存在しません: {enroll_dir}")
            return valid_speakers
        
        for speaker_id in os.listdir(enroll_dir):
            speaker_path = os.path.join(enroll_dir, speaker_id)
            
            if not os.path.isdir(speaker_path):
                continue
            
            if self.should_exclude_speaker(speaker_id):
                self.logger.info(f"背景話者として除外: {speaker_id}")
                continue
            
            # 音声ファイルが存在するかチェック
            audio_files = self._get_audio_files(speaker_path)
            if not audio_files:
                self.logger.warning(f"音声ファイルが見つかりません: {speaker_id}")
                continue
            
            valid_speakers.append(speaker_id)
            self.logger.info(f"有効な話者として登録: {speaker_id} ({len(audio_files)}ファイル)")
        
        return sorted(valid_speakers)
    
    def _get_audio_files(self, directory: str) -> List[str]:
        """
        ディレクトリ内の音声ファイルを取得
        
        Args:
            directory: 検索対象ディレクトリ
            
        Returns:
            音声ファイルパスのリスト
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        return audio_files
    
    def get_background_dataset_info(self) -> Dict[str, Any]:
        """
        背景データセットの情報を取得
        
        Returns:
            背景データセット情報の辞書
        """
        return {
            "jvs_speakers_count": len(self.jvs_speaker_ids),
            "jvs_speaker_ids": sorted(list(self.jvs_speaker_ids)),
            "common_voice_patterns": self.cv_patterns,
            "exclusion_enabled": self.exclude_background
        }
    
    def get_speaker_statistics(self, enroll_dir: str) -> Dict[str, Any]:
        """
        話者統計情報を取得
        
        Args:
            enroll_dir: 登録用ディレクトリパス
            
        Returns:
            統計情報の辞書
        """
        if not os.path.exists(enroll_dir):
            return {
                "total_speakers": 0,
                "valid_speakers": 0,
                "excluded_speakers": 0,
                "speakers_with_no_audio": 0
            }
        
        total_speakers = 0
        valid_speakers = 0
        excluded_speakers = 0
        speakers_with_no_audio = 0
        
        for speaker_id in os.listdir(enroll_dir):
            speaker_path = os.path.join(enroll_dir, speaker_id)
            
            if not os.path.isdir(speaker_path):
                continue
            
            total_speakers += 1
            
            if self.should_exclude_speaker(speaker_id):
                excluded_speakers += 1
                continue
            
            audio_files = self._get_audio_files(speaker_path)
            if not audio_files:
                speakers_with_no_audio += 1
                continue
            
            valid_speakers += 1
        
        return {
            "total_speakers": total_speakers,
            "valid_speakers": valid_speakers,
            "excluded_speakers": excluded_speakers,
            "speakers_with_no_audio": speakers_with_no_audio
        }

if __name__ == "__main__":
    # テスト実行
    manager = DatasetManager()
    
    # テスト用話者ID
    test_speakers = [
        "yamada_taro",
        "jvs001",
        "cv_speaker_123",
        "commonvoice_ja_001",
        "sato_hanako"
    ]
    
    print("=== 話者ID判定テスト ===")
    for speaker_id in test_speakers:
        is_jvs = manager.is_jvs_speaker(speaker_id)
        is_cv = manager.is_common_voice_speaker(speaker_id)
        is_background = manager.is_background_speaker(speaker_id)
        should_exclude = manager.should_exclude_speaker(speaker_id)
        
        print(f"話者ID: {speaker_id}")
        print(f"  JVS: {is_jvs}, CV: {is_cv}, 背景: {is_background}, 除外: {should_exclude}")
    
    print("\n=== 背景データセット情報 ===")
    bg_info = manager.get_background_dataset_info()
    print(f"JVS話者数: {bg_info['jvs_speakers_count']}")
    print(f"Common Voiceパターン: {bg_info['common_voice_patterns']}")
    print(f"除外設定: {bg_info['exclusion_enabled']}")