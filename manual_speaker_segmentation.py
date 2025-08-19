"""
手動話者セグメント分離システム
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import csv
from pathlib import Path
import tempfile
import os

from enhanced_speaker_recognition import JapaneseSpeakerRecognizer, RecognitionResult


@dataclass
class AudioSegment:
    """音声セグメント情報"""
    id: int
    start_time: float
    end_time: float
    duration: float
    assigned_speaker: Optional[str] = None
    recognition_results: Optional[List[Dict[str, Any]]] = None
    confidence_level: Optional[str] = None  # "高", "中", "低"
    
    @property
    def is_recognized(self) -> bool:
        """認識済みかどうか"""
        return self.recognition_results is not None
    
    @property
    def top_speaker(self) -> Optional[str]:
        """最上位話者候補"""
        if self.recognition_results and len(self.recognition_results) > 0:
            return self.recognition_results[0].get('speaker')
        return None
    
    @property
    def top_score(self) -> Optional[float]:
        """最上位スコア"""
        if self.recognition_results and len(self.recognition_results) > 0:
            return self.recognition_results[0].get('score')
        return None


class ManualSpeakerSegmentator:
    """手動話者セグメント分離クラス"""
    
    def __init__(self, recognizer: JapaneseSpeakerRecognizer):
        self.recognizer = recognizer
        self.audio_data = None
        self.sample_rate = None
        self.audio_duration = None
        self.segments: Dict[int, AudioSegment] = {}
        self.next_segment_id = 1
        
    def load_audio(self, audio_file_path: str) -> Tuple[bool, str]:
        """音声ファイルの読み込み"""
        try:
            # librosaで音声データを読み込み
            self.audio_data, self.sample_rate = librosa.load(
                audio_file_path, 
                sr=None,  # 元のサンプリングレートを保持
                mono=True
            )
            self.audio_duration = len(self.audio_data) / self.sample_rate
            
            # セグメント情報をリセット
            self.segments.clear()
            self.next_segment_id = 1
            
            return True, f"音声読み込み完了: {self.audio_duration:.2f}秒"
            
        except Exception as e:
            return False, f"音声読み込みエラー: {str(e)}"
    
    def save_audio_for_playback(self, output_path: str) -> Tuple[bool, str]:
        """再生用音声ファイルの保存"""
        try:
            if self.audio_data is None:
                return False, "音声データが読み込まれていません"
            
            # 音声データを再生用ファイルとして保存
            sf.write(output_path, self.audio_data, self.sample_rate)
            return True, f"再生用ファイル保存完了: {output_path}"
            
        except Exception as e:
            return False, f"再生用ファイル保存エラー: {str(e)}"
    
    def create_segment_audio_file(self, segment: AudioSegment, output_path: str) -> Tuple[bool, str]:
        """セグメント音声ファイルの作成"""
        try:
            segment_audio = self.extract_segment_audio(segment)
            sf.write(output_path, segment_audio, self.sample_rate)
            return True, f"セグメント音声ファイル作成完了: {output_path}"
            
        except Exception as e:
            return False, f"セグメント音声ファイル作成エラー: {str(e)}"
    
    def get_waveform_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """波形表示用のデータを取得"""
        if self.audio_data is None:
            raise ValueError("音声データが読み込まれていません")
        
        # 時間軸を生成
        time_axis = np.linspace(0, self.audio_duration, len(self.audio_data))
        
        return time_axis, self.audio_data, self.sample_rate
    
    def create_segment(self, start_time: float, end_time: float) -> Tuple[bool, str, Optional[AudioSegment]]:
        """セグメントの作成"""
        try:
            # 入力値の検証
            if start_time >= end_time:
                return False, "開始時間は終了時間より前である必要があります", None
            
            if start_time < 0 or end_time > self.audio_duration:
                return False, f"時間は0秒から{self.audio_duration:.2f}秒の間で指定してください", None
            
            duration = end_time - start_time
            if duration < 0.5:
                return False, "セグメントは最低0.5秒以上である必要があります", None
            
            # 重複チェック
            for segment in self.segments.values():
                if not (end_time <= segment.start_time or start_time >= segment.end_time):
                    return False, f"セグメント{segment.id}と時間が重複しています", None
            
            # セグメント作成
            segment = AudioSegment(
                id=self.next_segment_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
            self.segments[self.next_segment_id] = segment
            self.next_segment_id += 1
            
            return True, f"セグメント{segment.id}を作成しました ({start_time:.1f}s-{end_time:.1f}s)", segment
            
        except Exception as e:
            return False, f"セグメント作成エラー: {str(e)}", None
    
    def delete_segment(self, segment_id: int) -> Tuple[bool, str]:
        """セグメントの削除"""
        if segment_id not in self.segments:
            return False, f"セグメント{segment_id}が見つかりません"
        
        del self.segments[segment_id]
        return True, f"セグメント{segment_id}を削除しました"
    
    def update_segment_times(self, segment_id: int, start_time: float, end_time: float) -> Tuple[bool, str]:
        """セグメント時間の更新"""
        if segment_id not in self.segments:
            return False, f"セグメント{segment_id}が見つかりません"
        
        # 一時的にセグメントを削除して重複チェック
        temp_segment = self.segments.pop(segment_id)
        
        # 重複チェック（自分以外のセグメント）
        for other_segment in self.segments.values():
            if not (end_time <= other_segment.start_time or start_time >= other_segment.end_time):
                # セグメントを元に戻す
                self.segments[segment_id] = temp_segment
                return False, f"セグメント{other_segment.id}と時間が重複しています"
        
        # セグメントを更新
        temp_segment.start_time = start_time
        temp_segment.end_time = end_time
        temp_segment.duration = end_time - start_time
        self.segments[segment_id] = temp_segment
        
        return True, f"セグメント{segment_id}を更新しました"
    
    def extract_segment_audio(self, segment: AudioSegment) -> np.ndarray:
        """セグメント音声の抽出"""
        if self.audio_data is None:
            raise ValueError("音声データが読み込まれていません")
        
        start_sample = int(segment.start_time * self.sample_rate)
        end_sample = int(segment.end_time * self.sample_rate)
        
        return self.audio_data[start_sample:end_sample]
    
    def recognize_all_segments(self, show_jvs: bool = False, show_cv: bool = False) -> Tuple[bool, str]:
        """全セグメントの認識実行"""
        if not self.segments:
            return False, "認識するセグメントがありません"
        
        try:
            recognized_count = 0
            
            for segment in self.segments.values():
                # セグメント音声を抽出
                segment_audio = self.extract_segment_audio(segment)
                
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, segment_audio, self.sample_rate)
                    temp_file_path = temp_file.name
                
                try:
                    # 認識実行（ファイルパスを使用）
                    result = self.recognizer.recognize_speaker(temp_file_path)
                    
                    if result and result.all_scores:
                        # all_scoresを使って結果をJVS/Common Voice設定に基づいてフィルタリング
                        filtered_scores = self._filter_recognition_results(
                            result.all_scores, show_jvs, show_cv
                        )
                        
                        # スコア順でソート
                        sorted_results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # Top-5結果を保存
                        segment.recognition_results = []
                        for i, (speaker, score) in enumerate(sorted_results[:5]):
                            segment.recognition_results.append({
                                'rank': i + 1,
                                'speaker': speaker,
                                'score': float(score)
                            })
                        
                        # 信頼度レベル設定（最上位スコアを使用）
                        top_score = sorted_results[0][1] if sorted_results else 0.0
                        if top_score >= 0.7:
                            segment.confidence_level = "高"
                        elif top_score >= 0.4:
                            segment.confidence_level = "中"
                        else:
                            segment.confidence_level = "低"
                        
                        recognized_count += 1
                    
                finally:
                    # 一時ファイル削除
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            return True, f"{recognized_count}/{len(self.segments)}個のセグメントを認識しました"
            
        except Exception as e:
            return False, f"認識エラー: {str(e)}"
    
    def _filter_recognition_results(self, all_scores: Dict[str, float], 
                                  show_jvs: bool, show_cv: bool) -> Dict[str, float]:
        """認識結果をJVS/Common Voice設定に基づいてフィルタリング"""
        filtered_scores = {}
        
        for speaker, score in all_scores.items():
            # JVS話者の判定
            is_jvs = speaker.startswith('jvs')
            is_cv = speaker.startswith(('cv_', 'commonvoice_'))
            
            # フィルタリング条件
            should_include = True
            if is_jvs and not show_jvs:
                should_include = False
            elif is_cv and not show_cv:
                should_include = False
            
            if should_include:
                filtered_scores[speaker] = score
        
        return filtered_scores
    
    def assign_speaker_to_segment(self, segment_id: int, speaker_name: str) -> Tuple[bool, str]:
        """セグメントに話者を割り当て"""
        if segment_id not in self.segments:
            return False, f"セグメント{segment_id}が見つかりません"
        
        self.segments[segment_id].assigned_speaker = speaker_name
        return True, f"セグメント{segment_id}に話者'{speaker_name}'を割り当てました"
    
    def get_segments_list(self) -> List[AudioSegment]:
        """セグメント一覧を取得（時間順）"""
        return sorted(self.segments.values(), key=lambda x: x.start_time)
    
    def get_timeline_data(self) -> Dict[str, Any]:
        """タイムライン表示用のデータを生成"""
        timeline_data = {
            'speakers': {},
            'total_duration': self.audio_duration,
            'segments': []
        }
        
        # 話者別にセグメントを分類
        for segment in self.segments.values():
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
    
    def export_to_csv(self, output_path: str) -> Tuple[bool, str]:
        """CSV形式でエクスポート"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['segment_id', 'start_time', 'end_time', 'duration', 
                             'assigned_speaker', 'top_speaker', 'top_score', 'confidence_level']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for segment in self.get_segments_list():
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
            
            return True, f"CSVファイルを保存しました: {output_path}"
            
        except Exception as e:
            return False, f"CSV保存エラー: {str(e)}"
    
    def export_to_json(self, output_path: str) -> Tuple[bool, str]:
        """JSON形式でエクスポート"""
        try:
            timeline_data = self.get_timeline_data()
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(timeline_data, jsonfile, indent=2, ensure_ascii=False)
            
            return True, f"JSONファイルを保存しました: {output_path}"
            
        except Exception as e:
            return False, f"JSON保存エラー: {str(e)}"
    
    def export_to_srt(self, output_path: str) -> Tuple[bool, str]:
        """SRT字幕形式でエクスポート"""
        try:
            def format_time(seconds):
                """SRT時間形式に変換"""
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
            
            with open(output_path, 'w', encoding='utf-8') as srtfile:
                for i, segment in enumerate(self.get_segments_list(), 1):
                    speaker = segment.assigned_speaker or segment.top_speaker or 'unknown'
                    start_time = format_time(segment.start_time)
                    end_time = format_time(segment.end_time)
                    
                    srtfile.write(f"{i}\n")
                    srtfile.write(f"{start_time} --> {end_time}\n")
                    srtfile.write(f"{speaker}\n\n")
            
            return True, f"SRTファイルを保存しました: {output_path}"
            
        except Exception as e:
            return False, f"SRT保存エラー: {str(e)}"