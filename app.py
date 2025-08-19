"""
日本語話者認識システム - Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import time
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from enhanced_speaker_recognition import JapaneseSpeakerRecognizer, RecognitionResult
from dataset_manager import DatasetManager
from manual_speaker_segmentation_v2 import ManualSpeakerSegmentatorV2
from segmentation_core import SegmentationConfig
from segmentation_core import AudioSegment

# ページ設定
st.set_page_config(
    page_title="日本語話者認識システム",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'speakers_enrolled' not in st.session_state:
    st.session_state.speakers_enrolled = 0

# 手動話者分離用のセッション状態
if 'manual_segmentator' not in st.session_state:
    st.session_state.manual_segmentator = None
if 'manual_audio_loaded' not in st.session_state:
    st.session_state.manual_audio_loaded = False
if 'manual_segments' not in st.session_state:
    st.session_state.manual_segments = {}
if 'manual_segment_selection' not in st.session_state:
    st.session_state.manual_segment_selection = {'start': None, 'end': None}
if 'manual_show_jvs' not in st.session_state:
    st.session_state.manual_show_jvs = False
if 'manual_show_cv' not in st.session_state:
    st.session_state.manual_show_cv = False
if 'input_reset_counter' not in st.session_state:
    st.session_state.input_reset_counter = 0
if 'manual_segment_start_time' not in st.session_state:
    st.session_state.manual_segment_start_time = 0.0
if 'manual_segment_end_time' not in st.session_state:
    st.session_state.manual_segment_end_time = 0.0

# 音声再生コントロール用のセッション状態
if 'audio_player_state' not in st.session_state:
    st.session_state.audio_player_state = {
        'is_playing': False,
        'current_position': 0.0,
        'play_start_time': 0.0,
        'play_end_time': None,
        'selection_start': None,
        'selection_end': None,
        'selection_mode': 'time_range',  # 'time_range', 'segment', 'full'
        'auto_play': True,
        'loop_play': False,
        'audio_file_path': None,
        'play_speed': 1.0,
        # Phase 1: 再生位置追跡用の新しいフィールド
        'playback_start_timestamp': None,  # 再生開始時刻（time.time()）
        'playback_paused_position': 0.0,   # 一時停止時の位置
        'show_playback_position': True,    # 再生位置線の表示フラグ
        'show_playback_range': True,       # 再生範囲の表示フラグ
        'manual_position': None,           # 手動設定された位置
        'last_update_time': None           # 最終更新時刻
    }

# 複数話者分析用のセッション状態
if 'multi_recognizer' not in st.session_state:
    st.session_state.multi_recognizer = None
if 'diarization_initialized' not in st.session_state:
    st.session_state.diarization_initialized = False
if 'diarization_show_jvs' not in st.session_state:
    st.session_state.diarization_show_jvs = False
if 'diarization_show_cv' not in st.session_state:
    st.session_state.diarization_show_cv = False

# 単一話者識別用のセッション状態
if 'show_jvs_in_results' not in st.session_state:
    st.session_state.show_jvs_in_results = False
if 'show_common_voice_in_results' not in st.session_state:
    st.session_state.show_common_voice_in_results = False

# メイン関数
def main():
    st.title("🎤 日本語話者認識システム")
    st.markdown("### SpeechBrain + ECAPA-TDNNによる高精度話者識別")
    
    # サイドバーでシステム制御
    setup_sidebar()
    
    # メインコンテンツ
    if st.session_state.model_loaded:
        display_main_content()
    else:
        display_welcome_page()

def setup_sidebar():
    """サイドバーの設定"""
    st.sidebar.header("🔧 システム制御")
    
    # モデル初期化
    if st.sidebar.button("🚀 モデル初期化", type="primary"):
        initialize_system()
    
    # システム状態表示
    st.sidebar.subheader("📊 システム状態")
    
    status_color = "🟢" if st.session_state.model_loaded else "🔴"
    st.sidebar.write(f"{status_color} モデル: {'読み込み済み' if st.session_state.model_loaded else '未読み込み'}")
    st.sidebar.write(f"👥 登録話者数: {st.session_state.speakers_enrolled}")
    
    if st.session_state.model_loaded:
        
        # 埋め込みキャッシュ管理
        st.sidebar.subheader("💾 埋め込みキャッシュ")
        
        # キャッシュ状態表示
        if st.session_state.recognizer:
            cache_exists = os.path.exists("enrolled_speakers_embeddings.npz")
            cache_color = "🟢" if cache_exists else "🔴"
            st.sidebar.write(f"{cache_color} キャッシュ: {'存在' if cache_exists else '未作成'}")
            
            # キャッシュ管理ボタン
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("💾 保存") and st.session_state.recognizer.speaker_embeddings:
                    if st.session_state.recognizer.save_speaker_embeddings():
                        st.sidebar.success("✅ 保存完了")
                    else:
                        st.sidebar.error("❌ 保存失敗")
            
            with col2:
                if st.button("🗑️ 削除") and cache_exists:
                    try:
                        os.remove("enrolled_speakers_embeddings.npz")
                        st.sidebar.success("✅ 削除完了")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ 削除失敗: {e}")
        
        
        # システム情報表示
        if st.sidebar.button("ℹ️ システム情報"):
            show_system_info()

def initialize_system():
    """システムの初期化"""
    with st.spinner("システムを初期化中..."):
        try:
            # Recognizer初期化
            recognizer = JapaneseSpeakerRecognizer()
            
            # プログレスバー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # モデル読み込み
            status_text.text("モデルを読み込み中...")
            progress_bar.progress(25)
            
            if not recognizer.initialize_model():
                st.error("❌ モデルの初期化に失敗しました")
                return
            
            # 話者データベース構築
            status_text.text("話者データベースを構築中...")
            progress_bar.progress(50)
            
            enrolled_count = recognizer.build_speaker_database()
            
            # 背景モデル構築
            status_text.text("背景モデルを構築中...")
            progress_bar.progress(75)
            
            recognizer.build_background_model()
            
            # 完了
            progress_bar.progress(100)
            status_text.text("初期化完了！")
            
            # セッション状態更新
            st.session_state.recognizer = recognizer
            st.session_state.model_loaded = True
            st.session_state.speakers_enrolled = enrolled_count
            
            st.success(f"✅ システム初期化完了！{enrolled_count}名の話者を登録しました")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 初期化エラー: {str(e)}")


def show_system_info():
    """システム情報の表示"""
    if st.session_state.recognizer is None:
        st.error("システムが初期化されていません")
        return
    
    info = st.session_state.recognizer.get_system_info()
    
    st.sidebar.subheader("🔍 詳細情報")
    for key, value in info.items():
        st.sidebar.write(f"**{key}**: {value}")

def display_welcome_page():
    """ウェルカムページの表示"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 システム概要
        
        このシステムは**SpeechBrain**と**ECAPA-TDNN**を使用した
        高精度な日本語話者認識システムです。
        
        ### 🚀 主な機能
        
        - **高精度認識**: ECAPA-TDNNによる最先端の話者埋め込み
        - **背景話者除外**: JVS・Common Voice話者の自動除外
        - **スコア正規化**: 背景モデルによる認識精度向上
        - **直感的UI**: Streamlitによる使いやすいインターface
        
        ### 📋 使用方法
        
        1. **サイドバー**の「🚀 モデル初期化」をクリック
        2. 音声ファイルをアップロードして話者識別
        3. 結果とスコア詳細を確認
        
        ### 📁 データ準備
        
        ```
        enroll/
        ├── yamada_taro/     # 話者1のファイル
        ├── sato_hanako/     # 話者2のファイル
        └── tanaka_jiro/     # 話者3のファイル
        ```
        
        ---
        
        **準備ができたら、サイドバーでシステムを初期化してください！**
        """)

def display_main_content():
    """メインコンテンツの表示"""
    # タブ設定
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎤 単一話者識別", 
        "🎭 複数話者分析", 
        "🔊 手動話者分離",
        "👥 話者管理", 
        "📊 統計情報"
    ])
    
    with tab1:
        display_recognition_tab()
    
    with tab2:
        display_diarization_tab()
    
    with tab3:
        display_manual_segmentation_tab()
    
    with tab4:
        display_speaker_management_tab()
    
    with tab5:
        display_statistics_tab()

def display_recognition_tab():
    """単一話者識別タブ"""
    st.header("🎤 単一話者識別")
    st.caption("1名の話者を登録済み話者から識別します")
    
    if st.session_state.speakers_enrolled == 0:
        st.warning("⚠️ 登録された話者がいません。enrollフォルダに音声ファイルを配置してください。")
        return
    
    # 表示設定
    st.subheader("🎛️ 表示設定")
    col1, col2 = st.columns(2)
    
    with col1:
        show_jvs = st.checkbox(
            "🗾 JVS話者を結果に表示", 
            value=st.session_state.recognizer.config["ui"]["show_jvs_in_results"] if st.session_state.recognizer else True,
            help="認識結果のトップ10にJVS話者を含めるかどうか"
        )
    
    with col2:
        show_cv = st.checkbox(
            "🌐 Common Voice話者を結果に表示",
            value=st.session_state.recognizer.config["ui"]["show_common_voice_in_results"] if st.session_state.recognizer else False,
            help="認識結果のトップ10にCommon Voice話者を含めるかどうか"
        )
    
    # 設定を一時的にセッション状態に保存
    if 'show_jvs_in_results' not in st.session_state:
        st.session_state.show_jvs_in_results = show_jvs
    if 'show_common_voice_in_results' not in st.session_state:
        st.session_state.show_common_voice_in_results = show_cv
    
    # 設定が変更された場合の処理
    if (st.session_state.show_jvs_in_results != show_jvs or 
        st.session_state.show_common_voice_in_results != show_cv):
        
        # セッション状態を更新
        st.session_state.show_jvs_in_results = show_jvs
        st.session_state.show_common_voice_in_results = show_cv
        
        # 設定ファイルも更新
        if st.session_state.recognizer:
            st.session_state.recognizer.config["ui"]["show_jvs_in_results"] = show_jvs
            st.session_state.recognizer.config["ui"]["show_common_voice_in_results"] = show_cv
            
            # 設定ファイルに保存
            try:
                import json
                with open("config.json", 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.recognizer.config, f, indent=2, ensure_ascii=False)
            except Exception as e:
                st.error(f"設定保存エラー: {e}")
    else:
        # 変更がない場合はセッション状態を更新するだけ
        st.session_state.show_jvs_in_results = show_jvs
        st.session_state.show_common_voice_in_results = show_cv
    
    st.divider()
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="対応形式: WAV, MP3, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 音声情報表示
            st.subheader("📄 ファイル情報")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ファイル名**: {uploaded_file.name}")
                st.write(f"**ファイルサイズ**: {uploaded_file.size / 1024:.1f} KB")
            
            with col2:
                # 音声プレーヤー
                st.audio(uploaded_file.getvalue())
            
            # 識別実行
            if st.button("🔍 話者識別開始", type="primary"):
                perform_recognition(tmp_path)
        
        finally:
            # 一時ファイル削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def perform_recognition(audio_path: str):
    """話者識別の実行"""
    with st.spinner("音声を解析中..."):
        try:
            # 進捗表示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("音声を前処理中...")
            progress_bar.progress(25)
            
            # 話者識別実行
            status_text.text("話者埋め込みを抽出中...")
            progress_bar.progress(50)
            
            result = st.session_state.recognizer.recognize_speaker(audio_path)
            
            status_text.text("スコアを計算中...")
            progress_bar.progress(75)
            
            if result is None:
                st.error("❌ 話者識別に失敗しました")
                return
            
            progress_bar.progress(100)
            status_text.text("識別完了！")
            
            # 結果表示
            display_recognition_result(result)
            
        except Exception as e:
            st.error(f"❌ 識別エラー: {str(e)}")

def display_recognition_result(result: RecognitionResult):
    """認識結果の表示"""
    st.subheader("🎯 識別結果")
    
    # 表示設定に基づいてスコアをフィルタリング
    show_jvs = getattr(st.session_state, 'show_jvs_in_results', True)
    show_cv = getattr(st.session_state, 'show_common_voice_in_results', False)
    
    # 全スコアからフィルタリング済みスコアを取得
    if result.all_scores:
        filtered_scores = st.session_state.recognizer.filter_scores_for_display(
            result.all_scores, show_jvs, show_cv
        )
        
        # フィルタリング後の最上位話者を決定
        filtered_best_speaker = None
        filtered_best_score = None
        if filtered_scores:
            filtered_best_speaker = max(filtered_scores, key=filtered_scores.get)
            filtered_best_score = filtered_scores[filtered_best_speaker]
    else:
        filtered_scores = {}
        filtered_best_speaker = None
        filtered_best_score = None
    
    # 表示する話者と信頼度を決定
    display_speaker = result.speaker_id
    display_confidence = result.confidence
    display_raw_score = result.raw_score
    
    # フィルタリング設定でオリジナルの最上位話者が除外される場合
    if filtered_best_speaker and filtered_best_speaker != result.speaker_id:
        # JVS話者が除外される場合の判定
        is_original_jvs = st.session_state.recognizer.dataset_manager.is_jvs_speaker(result.speaker_id)
        is_original_cv = st.session_state.recognizer.dataset_manager.is_common_voice_speaker(result.speaker_id)
        
        if (is_original_jvs and not show_jvs) or (is_original_cv and not show_cv):
            display_speaker = filtered_best_speaker
            display_raw_score = filtered_best_score
            # 信頼度は同じロジックで計算（簡略化）
            display_confidence = filtered_best_score
    
    # メイン結果
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="識別された話者",
            value=display_speaker,
            delta=None
        )
    
    with col2:
        confidence_color = "🟢" if display_confidence > 0.5 else "🟡" if display_confidence > 0.25 else "🔴"
        st.metric(
            label="信頼度",
            value=f"{display_confidence:.3f}",
            delta=confidence_color
        )
    
    with col3:
        st.metric(
            label="生スコア",
            value=f"{display_raw_score:.3f}"
        )
    
    # しきい値チェック
    threshold = st.session_state.recognizer.threshold
    if display_confidence > threshold:
        st.success(f"✅ 信頼度がしきい値({threshold:.3f})を上回りました")
    else:
        st.warning(f"⚠️ 信頼度がしきい値({threshold:.3f})を下回りました")
    
    # 詳細スコア表示（トップ10）
    if filtered_scores:
        st.subheader("📊 トップ10話者スコア")
        filter_info = []
        if not show_jvs:
            filter_info.append("JVS話者を除外")
        if not show_cv:
            filter_info.append("Common Voice話者を除外")
        
        caption = f"上位{len(filtered_scores)}名の類似度スコア"
        if filter_info:
            caption += f" ({', '.join(filter_info)})"
        st.caption(caption)
        
        display_score_chart(filtered_scores, display_speaker)
    
    # 正規化スコア情報
    if result.normalized_score is not None:
        st.subheader("🔧 スコア正規化情報")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**正規化前**: {result.raw_score:.3f}")
            st.write(f"**正規化後**: {result.normalized_score:.3f}")
        
        with col2:
            improvement = result.normalized_score - result.raw_score
            improvement_color = "🟢" if improvement > 0 else "🔴"
            st.write(f"**改善度**: {improvement_color} {improvement:+.3f}")

def display_score_chart(scores: Dict[str, float], best_speaker: str):
    """スコアチャートの表示"""
    # データ準備
    speakers = list(scores.keys())
    score_values = list(scores.values())
    colors = ['red' if speaker == best_speaker else 'lightblue' for speaker in speakers]
    
    # 横棒グラフ
    fig = go.Figure(data=[
        go.Bar(
            y=speakers,
            x=score_values,
            orientation='h',
            marker_color=colors,
            text=[f"{score:.3f}" for score in score_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="話者別類似度スコア",
        xaxis_title="類似度スコア",
        yaxis_title="話者ID",
        height=max(300, len(speakers) * 40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_speaker_management_tab():
    """話者管理タブ"""
    st.header("👥 話者管理")
    
    # データセット管理
    dataset_manager = DatasetManager()
    
    # 統計情報
    stats = dataset_manager.get_speaker_statistics("enroll")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総話者数", stats["total_speakers"])
    
    with col2:
        st.metric("有効話者数", stats["valid_speakers"])
    
    with col3:
        st.metric("除外話者数", stats["excluded_speakers"])
    
    with col4:
        st.metric("音声なし話者数", stats["speakers_with_no_audio"])
    
    # 話者リスト表示
    if os.path.exists("enroll"):
        st.subheader("📋 話者一覧")
        
        speaker_data = []
        for speaker_id in os.listdir("enroll"):
            speaker_path = os.path.join("enroll", speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            
            # 音声ファイル数をカウント
            audio_files = dataset_manager._get_audio_files(speaker_path)
            
            # 状態判定
            if dataset_manager.should_exclude_speaker(speaker_id):
                status = "❌ 除外"
                reason = "背景話者"
            elif not audio_files:
                status = "⚠️ 音声なし"
                reason = "音声ファイルなし"
            else:
                status = "✅ 有効"
                reason = "登録可能"
            
            speaker_data.append({
                "話者ID": speaker_id,
                "音声ファイル数": len(audio_files),
                "状態": status,
                "理由": reason
            })
        
        if speaker_data:
            df = pd.DataFrame(speaker_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("📁 enrollフォルダに話者データが見つかりません")
    
    # 背景データセット情報
    st.subheader("🗂️ 背景データセット情報")
    bg_info = dataset_manager.get_background_dataset_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**JVS話者数**: {bg_info['jvs_speakers_count']}")
        st.write(f"**除外設定**: {'有効' if bg_info['exclusion_enabled'] else '無効'}")
    
    with col2:
        st.write(f"**Common Voiceパターン**: {', '.join(bg_info['common_voice_patterns'])}")

def display_statistics_tab():
    """統計情報タブ"""
    st.header("📊 システム統計情報")
    
    if st.session_state.recognizer is None:
        st.warning("システムが初期化されていません")
        return
    
    # システム情報
    info = st.session_state.recognizer.get_system_info()
    
    # メトリクス表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("登録話者数", info["enrolled_speakers"])
        st.metric("サンプリングレート", f"{info['sample_rate']} Hz")
    
    with col2:
        st.metric("背景サンプル数", info["background_samples"])
        st.metric("しきい値", f"{info['threshold']:.3f}")
    
    with col3:
        st.metric("デバイス", info["device"])
        st.metric("スコア正規化", "有効" if info["score_normalization"] else "無効")
    
    # 詳細情報
    st.subheader("🔧 詳細設定")
    
    details_data = {
        "項目": [
            "モデル名",
            "最小音声長",
            "最大音声長",
            "デバイス",
            "スコア正規化",
            "背景サンプル数"
        ],
        "値": [
            info["model_name"],
            f"{info['min_duration']} 秒",
            f"{info['max_duration']} 秒",
            info["device"],
            "有効" if info["score_normalization"] else "無効",
            f"{info['background_samples']} サンプル"
        ]
    }
    
    details_df = pd.DataFrame(details_data)
    st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    # パフォーマンス情報
    st.subheader("⚡ パフォーマンス情報")
    
    if info["device"] == "cuda":
        st.success("🚀 GPU加速が有効です")
    elif info["device"] == "mps":
        st.success("🍎 Apple Silicon GPU加速が有効です")
    else:
        st.info("💻 CPU処理で動作中です")

def display_diarization_tab():
    """複数話者分析タブ"""
    st.header("🎭 複数話者分析")
    st.caption("複数話者の音声から時系列での話者認識を行います")
    
    # 初期化確認は既にグローバルで実行済み
    
    # 初期化ボタン
    if not st.session_state.diarization_initialized:
        if st.button("🚀 複数話者分析システムを初期化", type="primary"):
            initialize_diarization_system()
        
        st.info("""
        💡 **複数話者分析について**
        
        このタブでは、複数の話者が同時に話している音声を分析し、以下を行います：
        
        📊 **ダイアライゼーション**: 「いつ誰が話しているか」を検出
        🎯 **話者識別**: 各時間帯の話者を登録済み話者から特定
        
        **必要な準備**:
        - Hugging Face Token が `.env` ファイルに設定済み
        - pyannote.audio の利用規約に同意済み
        """)
        return
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "複数話者の音声ファイルをアップロード",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="2名以上の話者が含まれる音声ファイル"
    )
    
    if uploaded_file:
        # 設定
        st.subheader("⚙️ 分析設定")
        
        # 話者数設定
        col1, col2 = st.columns(2)
        with col1:
            min_speakers = st.number_input("最小話者数", 1, 10, 1, help="音声に含まれる最小話者数")
        with col2:
            max_speakers = st.number_input("最大話者数", 1, 10, 5, help="音声に含まれる最大話者数")
        
        # 表示設定
        st.write("**認識結果の表示設定**")
        col1, col2 = st.columns(2)
        
        with col1:
            show_jvs = st.checkbox(
                "🗾 JVS話者を結果に表示", 
                value=st.session_state.diarization_show_jvs,
                help="認識結果にJVS (Japanese Versatile Speech) コーパスの話者を含めるかどうか"
            )
            st.session_state.diarization_show_jvs = show_jvs
        
        with col2:
            show_cv = st.checkbox(
                "🌐 Common Voice話者を結果に表示",
                value=st.session_state.diarization_show_cv,
                help="認識結果にMozilla Common Voiceの話者を含めるかどうか"
            )
            st.session_state.diarization_show_cv = show_cv
        
        # 音声情報表示
        st.subheader("📄 ファイル情報")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ファイル名**: {uploaded_file.name}")
            st.write(f"**ファイルサイズ**: {uploaded_file.size / 1024:.1f} KB")
        
        with col2:
            # 音声プレーヤー
            st.audio(uploaded_file.getvalue())
        
        # 分析実行
        if st.button("🎭 複数話者分析開始", type="primary"):
            perform_multi_speaker_analysis(uploaded_file, min_speakers, max_speakers, show_jvs, show_cv)

def initialize_diarization_system():
    """複数話者分析システム初期化"""
    with st.spinner("複数話者分析システムを初期化中..."):
        try:
            # 進捗表示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("モジュールを読み込み中...")
            progress_bar.progress(25)
            
            from speaker_diarization import MultiSpeakerRecognizer
            
            status_text.text("ダイアライゼーションモデルを初期化中...")
            progress_bar.progress(50)
            
            recognizer = MultiSpeakerRecognizer()
            
            status_text.text("話者認識システムを統合中...")
            progress_bar.progress(75)
            
            if recognizer.initialize():
                st.session_state.multi_recognizer = recognizer
                st.session_state.diarization_initialized = True
                
                progress_bar.progress(100)
                status_text.text("初期化完了！")
                
                st.success("✅ 複数話者分析システム初期化完了")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ 初期化に失敗しました。ログを確認してください。")
                
        except Exception as e:
            st.error(f"❌ 初期化エラー: {e}")
            st.info("💡 Hugging Face Token の設定や pyannote.audio の利用規約同意を確認してください")

def perform_multi_speaker_analysis(uploaded_file, min_speakers, max_speakers, show_jvs=True, show_cv=False):
    """複数話者分析実行"""
    with st.spinner("複数話者分析中..."):
        # 一時ファイル保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # 進捗表示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("ダイアライゼーション実行中...")
            progress_bar.progress(30)
            
            # 分析実行
            result = st.session_state.multi_recognizer.process_audio(
                tmp_path, min_speakers, max_speakers
            )
            
            status_text.text("話者認識実行中...")
            progress_bar.progress(70)
            
            progress_bar.progress(100)
            status_text.text("分析完了！")
            
            # 結果表示
            display_multi_speaker_result(result, show_jvs, show_cv)
            
        except Exception as e:
            st.error(f"❌ 分析エラー: {e}")
        finally:
            # 一時ファイル削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def display_multi_speaker_result(result, show_jvs=True, show_cv=False):
    """複数話者分析結果表示"""
    st.subheader("🎯 分析結果")
    
    # サマリー
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("検出話者数", result.total_speakers)
    with col2:
        st.metric("総時間", f"{result.total_duration:.1f}秒")
    with col3:
        # フィルタリング後のセグメント数も表示
        if len(result.segments) > 0:
            # フィルタリング計算
            filtered_count = 0
            for segment in result.segments:
                speaker = segment['recognized_speaker']
                if speaker.startswith('jvs') and not show_jvs:
                    continue
                if (speaker.startswith('cv_') or speaker.startswith('commonvoice_')) and not show_cv:
                    continue
                filtered_count += 1
            
            if filtered_count != len(result.segments):
                st.metric("表示セグメント数", f"{filtered_count}/{len(result.segments)}")
            else:
                st.metric("セグメント数", len(result.segments))
        else:
            st.metric("セグメント数", 0)
    
    # セグメント詳細
    if result.segments:
        # JVS/Common Voice話者のフィルタリング
        filtered_segments = []
        for segment in result.segments:
            speaker = segment['recognized_speaker']
            
            # JVS話者のチェック
            if speaker.startswith('jvs') and not show_jvs:
                continue
            
            # Common Voice話者のチェック  
            if (speaker.startswith('cv_') or speaker.startswith('commonvoice_')) and not show_cv:
                continue
            
            filtered_segments.append(segment)
        
        # フィルタリング情報の表示
        filter_info = []
        if not show_jvs:
            filter_info.append("JVS話者を除外")
        if not show_cv:
            filter_info.append("Common Voice話者を除外")
        
        if filter_info:
            st.caption(f"表示設定: {', '.join(filter_info)}")
        
        st.subheader("📋 時系列セグメント")
        
        if not filtered_segments:
            st.warning("⚠️ 表示設定により、すべてのセグメントが除外されました。表示設定を調整してください。")
            return
        
        for i, segment in enumerate(filtered_segments):
            # 認識成功のセグメントは緑、失敗は赤で表示
            if segment['recognized_speaker'] != "未認識":
                status_color = "🟢"
                confidence_text = f" (信頼度: {segment['confidence']:.3f})" if segment['confidence'] is not None else " (信頼度: N/A)"
            else:
                status_color = "🔴"
                confidence_text = ""
            
            with st.expander(
                f"{status_color} セグメント {segment['segment_id']}: "
                f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s "
                f"→ {segment['recognized_speaker']}{confidence_text}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**開始時間**: {segment['start_time']:.1f}秒")
                    st.write(f"**終了時間**: {segment['end_time']:.1f}秒")
                    st.write(f"**時間長**: {segment['duration']:.1f}秒")
                with col2:
                    st.write(f"**ダイアライゼーションラベル**: {segment['diarization_label']}")
                    st.write(f"**認識話者**: {segment['recognized_speaker']}")
                    if segment['confidence'] > 0:
                        confidence_value = segment['confidence']
                        st.write(f"**信頼度**: {confidence_value:.3f}" if confidence_value is not None else "**信頼度**: N/A")
                
                # トップ5話者スコア表示
                if 'all_scores' in segment and segment['all_scores']:
                    display_segment_top5_scores(segment, show_jvs, show_cv)
        
        # 📊 視覚化セクション
        st.subheader("📊 視覚化")
        
        # タブ構成
        tab1, tab2 = st.tabs([
            "⏰ ダイアライゼーション", 
            "👥 話者別タイムライン"
        ])
        
        with tab1:
            st.caption("pyannote.audioによる話者分離結果")
            display_diarization_timeline_chart(filtered_segments)
        
        with tab2:
            st.caption("話者認識結果ベースのタイムライン + 統計")
            display_speaker_summary_with_timeline(filtered_segments)
    
    else:
        st.warning("⚠️ セグメントが検出されませんでした。音声ファイルや設定を確認してください。")

def display_diarization_timeline_chart(segments):
    """ダイアライゼーションタイムラインチャート表示"""
    import plotly.graph_objects as go
    import plotly.colors as pc
    
    if not segments:
        st.warning("表示するセグメントがありません")
        return
    
    # ダイアライゼーションラベルリストと色の割り当て
    diarization_labels = list(set([s['diarization_label'] for s in segments]))
    diarization_labels.sort()  # 一貫した順序
    colors = pc.qualitative.Set3[:len(diarization_labels)]
    label_colors = dict(zip(diarization_labels, colors))
    
    fig = go.Figure()
    
    # 各セグメントをGanttチャートとして追加
    for segment in segments:
        diarization_label = segment['diarization_label']
        
        # ホバー情報
        hover_text = (
            f"ダイアライゼーションラベル: {diarization_label}<br>"
            f"認識話者: {segment['recognized_speaker']}<br>"
            f"時間: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s<br>"
            f"時間長: {segment['duration']:.1f}s<br>"
            f"信頼度: {format(segment['confidence'], '.3f') if segment['confidence'] is not None else 'N/A'}"
        )
        
        fig.add_trace(go.Bar(
            x=[segment['duration']],
            y=[diarization_label],
            base=segment['start_time'],
            orientation='h',
            name=f"{diarization_label} → {segment['recognized_speaker']}",
            marker_color=label_colors[diarization_label],
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    # レイアウト設定
    fig.update_layout(
        title="⏰ ダイアライゼーションタイムライン",
        xaxis_title="時間（秒）",
        yaxis_title="ダイアライゼーションラベル",
        height=max(300, len(diarization_labels) * 60),
        showlegend=False,
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_speaker_summary_with_timeline(segments):
    """話者別タイムライン + 統計情報の統合表示"""
    import plotly.graph_objects as go
    import plotly.colors as pc
    
    if not segments:
        st.warning("表示するセグメントがありません")
        return
    
    # 話者別タイムラインチャート
    speakers = list(set([s['recognized_speaker'] for s in segments]))
    speakers.sort()  # 一貫した順序
    colors = pc.qualitative.Set2[:len(speakers)]
    speaker_colors = dict(zip(speakers, colors))
    
    fig = go.Figure()
    
    # 各セグメントをGanttチャートとして追加
    for segment in segments:
        speaker = segment['recognized_speaker']
        
        # ホバー情報
        hover_text = (
            f"認識話者: {speaker}<br>"
            f"時間: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s<br>"
            f"時間長: {segment['duration']:.1f}s<br>"
            f"信頼度: {format(segment['confidence'], '.3f') if segment['confidence'] is not None else 'N/A'}<br>"
            f"元ラベル: {segment['diarization_label']}"
        )
        
        fig.add_trace(go.Bar(
            x=[segment['duration']],
            y=[speaker],
            base=segment['start_time'],
            orientation='h',
            name=speaker,
            marker_color=speaker_colors[speaker],
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    # レイアウト設定
    fig.update_layout(
        title="👥 話者別発話タイムライン",
        xaxis_title="時間（秒）",
        yaxis_title="認識話者",
        height=max(300, len(speakers) * 60),
        showlegend=False,
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 統計情報を計算・表示
    speaker_stats = calculate_speaker_statistics(segments)
    display_speaker_statistics_table(speaker_stats, speaker_colors, segments)

def calculate_speaker_statistics(segments):
    """話者別統計計算"""
    speaker_stats = {}
    
    for segment in segments:
        speaker = segment['recognized_speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'segments': 0,
                'total_time': 0.0,
                'total_confidence': 0.0,
                'avg_confidence': 0.0
            }
        
        speaker_stats[speaker]['segments'] += 1
        speaker_stats[speaker]['total_time'] += segment['duration']
        if segment['confidence'] > 0:
            speaker_stats[speaker]['total_confidence'] += segment['confidence']
    
    # 平均信頼度計算
    for speaker in speaker_stats:
        if speaker_stats[speaker]['segments'] > 0:
            speaker_stats[speaker]['avg_confidence'] = (
                speaker_stats[speaker]['total_confidence'] / 
                speaker_stats[speaker]['segments']
            )
    
    return speaker_stats

def display_speaker_statistics_table(speaker_stats, speaker_colors, segments):
    """話者別統計テーブル表示"""
    st.subheader("📊 話者別統計")
    
    if not speaker_stats:
        st.warning("表示する統計がありません")
        return
    
    # 合計時間計算
    total_time = sum([s['duration'] for s in segments])
    
    # テーブルヘッダー
    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        st.write("**話者**")
    with cols[1]:
        st.write("**セグメント数**")
    with cols[2]:
        st.write("**合計時間**")
    with cols[3]:
        st.write("**時間割合**")
    with cols[4]:
        st.write("**平均信頼度**")
    
    st.divider()
    
    # 各話者の統計を合計時間順で表示
    sorted_speakers = sorted(
        speaker_stats.items(), 
        key=lambda x: x[1]['total_time'], 
        reverse=True
    )
    
    for speaker, stats in sorted_speakers:
        time_ratio = (stats['total_time'] / total_time * 100) if total_time > 0 else 0
        
        cols = st.columns([3, 1, 1, 1, 1])
        with cols[0]:
            # 話者名に色インジケーター付き
            color = speaker_colors.get(speaker, "#888888")
            st.markdown(f'<span style="color: {color};">●</span> **{speaker}**', unsafe_allow_html=True)
        with cols[1]:
            st.write(f"{stats['segments']}")
        with cols[2]:
            st.write(f"{stats['total_time']:.1f}秒")
        with cols[3]:
            st.write(f"{time_ratio:.1f}%")
        with cols[4]:
            if stats['avg_confidence'] > 0:
                st.write(f"{stats['avg_confidence']:.3f}")
            else:
                st.write("N/A")

def display_segment_top5_scores(segment, show_jvs=True, show_cv=False):
    """セグメントのトップ5話者スコア表示"""
    all_scores = segment['all_scores']
    
    if not all_scores:
        return
    
    # フィルタリング適用
    filtered_scores = {}
    for speaker, score in all_scores.items():
        # JVS話者のチェック
        if speaker.startswith('jvs') and not show_jvs:
            continue
        
        # Common Voice話者のチェック  
        if (speaker.startswith('cv_') or speaker.startswith('commonvoice_')) and not show_cv:
            continue
        
        filtered_scores[speaker] = score
    
    if not filtered_scores:
        return
    
    # トップ5を取得
    sorted_scores = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
    top5_scores = sorted_scores[:5]
    
    st.divider()
    st.write("**🏆 トップ5話者スコア**")
    
    # フィルタリング情報
    filter_info = []
    if not show_jvs:
        filter_info.append("JVS話者を除外")
    if not show_cv:
        filter_info.append("Common Voice話者を除外")
    
    if filter_info:
        st.caption(f"表示設定: {', '.join(filter_info)}")
    
    # スコア表示
    for i, (speaker, score) in enumerate(top5_scores):
        rank = i + 1
        
        # 1位は太字、認識された話者は背景色付き
        if speaker == segment['recognized_speaker']:
            st.markdown(f"**{rank}. 🥇 {speaker}**: `{score:.3f}` ← **認識結果**")
        elif rank == 1:
            st.markdown(f"**{rank}. {speaker}**: **`{score:.3f}`**")
        else:
            medal = "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            st.write(f"{medal} {speaker}: `{score:.3f}`")
    
    # 表示された結果数を表示
    if len(filtered_scores) > 5:
        st.caption(f"他 {len(filtered_scores) - 5} 名の候補")

def display_manual_segmentation_tab():
    """手動話者分離タブ"""
    st.header("🔊 手動話者分離")
    st.caption("音声波形を見ながら手動でセグメントを作成し、話者識別を行います")
    
    if st.session_state.speakers_enrolled == 0:
        st.warning("⚠️ 登録された話者がいません。enrollフォルダに音声ファイルを配置してください。")
        return
    
    # JVS/Common Voice表示設定
    st.subheader("🎛️ 認識設定")
    col1, col2 = st.columns(2)
    
    with col1:
        show_jvs = st.checkbox(
            "🗾 JVS話者を結果に表示",
            value=st.session_state.recognizer.config["ui"]["show_jvs_in_results"] if st.session_state.recognizer else False,
            help="認識結果のTop-5候補にJVS話者を含めるかどうか"
        )
    
    with col2:
        show_cv = st.checkbox(
            "🌐 Common Voice話者を結果に表示",
            value=st.session_state.recognizer.config["ui"]["show_common_voice_in_results"] if st.session_state.recognizer else False,
            help="認識結果のTop-5候補にCommon Voice話者を含めるかどうか"
        )
    
    # セッション状態に保存
    if 'manual_show_jvs' not in st.session_state:
        st.session_state.manual_show_jvs = show_jvs
    if 'manual_show_cv' not in st.session_state:
        st.session_state.manual_show_cv = show_cv
    
    st.session_state.manual_show_jvs = show_jvs
    st.session_state.manual_show_cv = show_cv
    
    # Step 1: 音声ファイルアップロード
    st.subheader("📁 Step 1: 音声ファイルアップロード")
    uploaded_file = st.file_uploader(
        "音声ファイルを選択してください",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        key="manual_audio_upload"
    )
    
    if uploaded_file:
        # セグメンテーターの初期化
        if st.session_state.manual_segmentator is None:
            if st.session_state.recognizer:
                # 設定の初期化
                config = SegmentationConfig()
                st.session_state.manual_segmentator = ManualSpeakerSegmentatorV2(st.session_state.recognizer, config)
            else:
                st.error("システムが初期化されていません。サイドバーでモデルを初期化してください。")
                return
        
        # 音声ファイル読み込み
        if not st.session_state.manual_audio_loaded:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension not in ['wav', 'mp3', 'flac', 'm4a', 'ogg']:
                    st.error(f"サポートされていないファイル形式です: {file_extension}")
                    return
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
            except Exception as e:
                st.error(f"ファイル処理エラー: {str(e)}")
                return
            
            with st.spinner("音声ファイルを読み込み中..."):
                result = st.session_state.manual_segmentator.load_audio(temp_file_path)
                
                if result.success:
                    st.session_state.manual_audio_loaded = True
                    st.success(result.message)
                else:
                    st.error(result.message)
                    return
                
                # 一時ファイル削除
                os.unlink(temp_file_path)
        
        if st.session_state.manual_audio_loaded:
            display_manual_segmentation_interface()
    else:
        if st.session_state.manual_audio_loaded:
            # ファイルが削除された場合のリセット
            st.session_state.manual_audio_loaded = False
            st.session_state.manual_segmentator = None
            st.session_state.manual_segments = {}

def display_manual_segmentation_interface():
    """手動セグメンテーションインターフェース"""
    segmentator = st.session_state.manual_segmentator
    
    # 再生用音声ファイルの準備
    if st.session_state.audio_player_state['audio_file_path'] is None:
        temp_audio_path = f"temp_audio_{int(time.time())}.wav"
        result = segmentator.save_audio_for_playback(temp_audio_path)
        success = result.success
        if success:
            st.session_state.audio_player_state['audio_file_path'] = temp_audio_path
    
    # Step 2: 音声再生コントロールと波形表示
    st.subheader("🎵 Step 2: 音声再生と波形表示")
    
    # 音声コントロールパネル
    display_audio_controls(segmentator)
    
    # インタラクティブ波形表示
    display_interactive_waveform(segmentator)
    
    # セグメント作成UI
    display_segment_creation_ui()
    
    # Step 3: セグメント一覧と管理
    segments = segmentator.get_segments_list()
    if segments:
        display_segment_management(segments)
        
        # Step 4: 認識実行
        display_recognition_execution()
        
        # Step 5: 結果表示
        display_manual_segmentation_results(segments)

# Phase 1: 再生位置管理のヘルパー関数
def update_current_playback_position():
    """現在の再生位置を計算・更新"""
    import time
    
    player_state = st.session_state.audio_player_state
    
    if not player_state['is_playing']:
        # 再生していない場合は一時停止位置を維持
        return player_state['playback_paused_position']
    
    if player_state['playback_start_timestamp'] is None:
        # 再生開始時刻が未設定の場合は開始位置
        return player_state['play_start_time']
    
    # 再生開始からの経過時間を計算
    current_time = time.time()
    elapsed_time = current_time - player_state['playback_start_timestamp']
    
    # 再生速度を考慮
    adjusted_elapsed = elapsed_time * player_state['play_speed']
    
    # 現在位置を計算
    current_pos = player_state['playback_paused_position'] + adjusted_elapsed
    
    # 再生範囲内に制限
    play_end = player_state['play_end_time']
    if play_end and current_pos >= play_end:
        # 再生終了
        player_state['is_playing'] = False
        player_state['playback_paused_position'] = play_end
        return play_end
    
    # 現在位置を更新
    player_state['current_position'] = current_pos
    player_state['last_update_time'] = current_time
    
    return current_pos

def set_playback_position(position):
    """再生位置を手動設定"""
    import time
    
    player_state = st.session_state.audio_player_state
    
    # 位置を範囲内に制限
    play_start = player_state['play_start_time']
    play_end = player_state['play_end_time']
    
    if play_end:
        position = max(play_start, min(position, play_end))
    else:
        position = max(play_start, position)
    
    # 位置を設定
    player_state['current_position'] = position
    player_state['playback_paused_position'] = position
    player_state['manual_position'] = position
    player_state['last_update_time'] = time.time()
    
    # 再生中の場合は新しい開始時刻を設定
    if player_state['is_playing']:
        player_state['playback_start_timestamp'] = time.time()

def set_segment_time_from_current_position(time_type: str):
    """現在の再生位置をセグメント時間に設定
    
    Args:
        time_type: 'start' または 'end'
    """
    # デバッグ用: 再生状態を確認
    player_state = st.session_state.audio_player_state
    current_pos = update_current_playback_position()
    
    # 現在の再生状態をログ出力
    st.write(f"🔍 デバッグ情報:")
    st.write(f"- 再生中: {player_state['is_playing']}")
    st.write(f"- 現在位置: {current_pos:.1f}秒")
    st.write(f"- 一時停止位置: {player_state['playback_paused_position']:.1f}秒")
    st.write(f"- 現在位置(current_position): {player_state['current_position']:.1f}秒")
    
    # より確実な現在位置取得
    if player_state['is_playing']:
        # 再生中の場合
        final_pos = current_pos
    else:
        # 停止中の場合は current_position を使用
        final_pos = player_state.get('current_position', 0.0)
    
    if time_type == 'start':
        st.session_state.manual_segment_start_time = final_pos
        st.success(f"✅ 開始時間を {final_pos:.1f}秒 に設定しました")
    elif time_type == 'end':
        st.session_state.manual_segment_end_time = final_pos
        st.success(f"✅ 終了時間を {final_pos:.1f}秒 に設定しました")
    
    st.session_state.input_reset_counter += 1
    st.rerun()

def start_playback_from_position(position=None):
    """指定位置から再生開始"""
    import time
    
    player_state = st.session_state.audio_player_state
    
    if position is not None:
        set_playback_position(position)
    
    # 再生状態を設定
    player_state['is_playing'] = True
    player_state['playback_start_timestamp'] = time.time()
    player_state['last_update_time'] = time.time()

def display_audio_controls(segmentator):
    """音声再生コントロールパネル"""
    st.markdown("#### 🎵 音声再生コントロール")
    
    player_state = st.session_state.audio_player_state
    
    # 再生コントロールボタン
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 2, 1])
    
    with col1:
        if st.button("▶️ 再生", key="play_button"):
            handle_play_button(segmentator)
    
    with col2:
        if st.button("⏸️ 一時停止", key="pause_button"):
            # Phase 1: 現在位置を保存して一時停止
            current_pos = update_current_playback_position()
            player_state['is_playing'] = False
            player_state['playback_paused_position'] = current_pos
            player_state['current_position'] = current_pos
    
    with col3:
        if st.button("⏹️ 停止", key="stop_button"):
            # Phase 1: 停止して開始位置にリセット
            player_state['is_playing'] = False
            player_state['playback_paused_position'] = player_state['play_start_time']
            player_state['current_position'] = player_state['play_start_time']
            player_state['manual_position'] = None
    
    with col4:
        # Phase 1: リアルタイム位置表示
        current_pos = update_current_playback_position()
        total_time = segmentator.audio_duration
        play_end = player_state['play_end_time'] or total_time
        
        st.write(f"**位置**: {current_pos:.1f}秒 / {play_end:.1f}秒")
        
        # プログレスバー（再生範囲に対する相対位置）
        play_start = player_state['play_start_time']
        play_duration = play_end - play_start
        relative_pos = (current_pos - play_start) / play_duration if play_duration > 0 else 0.0
        progress = max(0.0, min(relative_pos, 1.0))
        st.progress(progress)
    
    with col5:
        # 再生状態インジケータ
        status_icon = "🔴" if player_state['is_playing'] else "⚪"
        status_text = "再生中" if player_state['is_playing'] else "停止中"
        st.write(f"{status_icon} {status_text}")
    
    # 再生設定
    st.markdown("#### ⚙️ 再生設定")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selection_mode = st.selectbox(
            "再生モード",
            options=["time_range", "segment", "full"],
            format_func=lambda x: {"time_range": "選択範囲", "segment": "セグメント別", "full": "全体"}[x],
            index=["time_range", "segment", "full"].index(player_state['selection_mode'])
        )
        player_state['selection_mode'] = selection_mode
    
    with col2:
        auto_play = st.checkbox("自動再生", value=player_state['auto_play'])
        player_state['auto_play'] = auto_play
    
    with col3:
        loop_play = st.checkbox("ループ再生", value=player_state['loop_play'])
        player_state['loop_play'] = loop_play
    
    # Phase 1: 手動位置設定機能
    st.markdown("#### 🎯 再生位置設定")
    col_pos1, col_pos2, col_pos3 = st.columns([2, 1, 1])
    
    with col_pos1:
        # 現在位置表示と手動設定
        current_pos = update_current_playback_position()
        play_start = player_state['play_start_time']
        play_end = player_state['play_end_time'] or segmentator.audio_duration
        
        new_position = st.slider(
            "再生位置 (秒)",
            min_value=float(play_start),
            max_value=float(play_end),
            value=float(current_pos),
            step=0.1,
            format="%.1f",
            key="position_slider"
        )
        
        # スライダーが変更された場合
        if abs(new_position - current_pos) > 0.05:  # 0.05秒以上の差がある場合
            set_playback_position(new_position)
    
    with col_pos2:
        if st.button("📍 位置設定", key="set_position"):
            position_input = st.number_input(
                "位置 (秒)",
                min_value=float(play_start),
                max_value=float(play_end),
                value=float(current_pos),
                step=0.1,
                key="manual_position_input"
            )
            set_playback_position(position_input)
    
    with col_pos3:
        if st.button("▶️ ここから再生", key="play_from_position"):
            start_playback_from_position()
    
    # 音声プレーヤー統合セクション
    display_integrated_audio_player(segmentator, player_state)

def display_integrated_audio_player(segmentator, player_state):
    """統合音声プレーヤー：波形表示と連動"""
    st.markdown("#### 🔊 統合音声プレーヤー")
    
    # プレーヤータブ
    tab1, tab2, tab3 = st.tabs(["🎵 全体音声", "📄 範囲音声", "🎯 設定"])
    
    with tab1:
        # 全体音声プレーヤー
        if player_state['audio_file_path'] and os.path.exists(player_state['audio_file_path']):
            st.markdown("**🌍 全体音声ファイル**")
            st.caption(f"音声長: {segmentator.audio_duration:.1f}秒")
            
            with open(player_state['audio_file_path'], 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/wav')
        else:
            st.warning("音声ファイルが準備中です...")
    
    with tab2:
        # 範囲音声プレーヤー
        display_range_audio_player(segmentator, player_state)
    
    with tab3:
        # プレーヤー設定
        display_player_settings(player_state)

def display_range_audio_player(segmentator, player_state):
    """範囲音声プレーヤー"""
    
    # 再生範囲または選択範囲の音声を生成
    range_start = None
    range_end = None
    range_label = ""
    
    if player_state['play_end_time'] and player_state['play_start_time'] < player_state['play_end_time']:
        # 再生範囲が設定されている場合
        range_start = player_state['play_start_time']
        range_end = player_state['play_end_time']
        range_label = f"再生範囲 ({range_start:.1f}s - {range_end:.1f}s)"
        range_color = "🔵"
    elif player_state['selection_start'] is not None and player_state['selection_end'] is not None:
        # 選択範囲がある場合
        range_start = player_state['selection_start']
        range_end = player_state['selection_end']
        range_label = f"選択範囲 ({range_start:.1f}s - {range_end:.1f}s)"
        range_color = "🟨"
    
    if range_start is not None and range_end is not None:
        range_duration = range_end - range_start
        st.markdown(f"**{range_color} {range_label}**")
        st.caption(f"範囲長: {range_duration:.1f}秒")
        
        # 範囲音声の自動チェックと生成
        cache_key = f"{range_start:.1f}_{range_end:.1f}"
        has_cached_audio = (
            'range_audio_cache' in st.session_state and 
            cache_key in st.session_state.range_audio_cache and
            os.path.exists(st.session_state.range_audio_cache[cache_key]['path'])
        )
        
        # 範囲音声ファイルを生成/表示
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if not has_cached_audio:
                if st.button("🎵 範囲音声を生成", key="generate_range_audio"):
                    generate_and_play_range_audio(segmentator, range_start, range_end, range_label)
            else:
                st.success("✅ 範囲音声が利用可能です")
        
        with col2:
            if st.button("🔄 再生成", key="refresh_range_audio"):
                # 該当範囲のキャッシュのみクリア
                clear_specific_range_cache(cache_key)
                generate_and_play_range_audio(segmentator, range_start, range_end, range_label)
        
        with col3:
            if has_cached_audio and st.button("🗑️ 削除", key="delete_range_audio"):
                clear_specific_range_cache(cache_key)
                st.success("削除完了")
                st.rerun()
        
        # 既存の範囲音声ファイルがあれば表示
        if has_cached_audio:
            display_cached_range_audio(range_start, range_end)
        
        # 自動生成のオプション
        if not has_cached_audio:
            if st.checkbox("🚀 自動生成（範囲変更時に自動で音声生成）", key="auto_generate_range"):
                generate_and_play_range_audio(segmentator, range_start, range_end, range_label)
        
    else:
        st.info("📝 **範囲未選択**\n\n再生範囲または選択範囲を設定すると、その部分の音声を再生できます。")

def generate_and_play_range_audio(segmentator, start_time, end_time, label):
    """範囲音声ファイルを生成して表示"""
    import tempfile
    import time
    
    try:
        # 一時ファイル名生成
        timestamp = int(time.time())
        temp_filename = f"range_audio_{start_time:.1f}_{end_time:.1f}_{timestamp}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # 音声データ取得
        time_axis, audio_data, sample_rate = segmentator.get_waveform_data()
        
        # 範囲の音声データを抽出
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        range_audio = audio_data[start_sample:end_sample]
        
        # 音声ファイルとして保存
        import soundfile as sf
        sf.write(temp_path, range_audio, sample_rate)
        
        # session_stateに保存
        if 'range_audio_cache' not in st.session_state:
            st.session_state.range_audio_cache = {}
        
        cache_key = f"{start_time:.1f}_{end_time:.1f}"
        st.session_state.range_audio_cache[cache_key] = {
            'path': temp_path,
            'label': label,
            'duration': end_time - start_time,
            'timestamp': timestamp
        }
        
        st.success(f"✅ {label} の音声ファイルを生成しました！")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ 音声生成エラー: {str(e)}")

def display_cached_range_audio(start_time, end_time):
    """キャッシュされた範囲音声ファイルを表示"""
    if 'range_audio_cache' not in st.session_state:
        return
    
    cache_key = f"{start_time:.1f}_{end_time:.1f}"
    if cache_key in st.session_state.range_audio_cache:
        cache_data = st.session_state.range_audio_cache[cache_key]
        
        if os.path.exists(cache_data['path']):
            st.markdown(f"**🎵 {cache_data['label']}**")
            st.caption(f"長さ: {cache_data['duration']:.1f}秒")
            
            with open(cache_data['path'], 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/wav')
        else:
            # ファイルが削除されている場合はキャッシュからも削除
            del st.session_state.range_audio_cache[cache_key]

def clear_range_audio_cache():
    """範囲音声キャッシュをクリア"""
    if 'range_audio_cache' in st.session_state:
        # ファイルを削除
        for cache_data in st.session_state.range_audio_cache.values():
            if os.path.exists(cache_data['path']):
                try:
                    os.unlink(cache_data['path'])
                except:
                    pass  # ファイル削除に失敗しても続行
        
        # キャッシュをクリア
        st.session_state.range_audio_cache.clear()

def clear_specific_range_cache(cache_key):
    """特定範囲のキャッシュのみクリア"""
    if 'range_audio_cache' in st.session_state and cache_key in st.session_state.range_audio_cache:
        cache_data = st.session_state.range_audio_cache[cache_key]
        
        # ファイルを削除
        if os.path.exists(cache_data['path']):
            try:
                os.unlink(cache_data['path'])
            except:
                pass  # ファイル削除に失敗しても続行
        
        # キャッシュから削除
        del st.session_state.range_audio_cache[cache_key]

def display_player_settings(player_state):
    """プレーヤー設定"""
    st.markdown("**⚙️ 表示設定**")
    
    # 表示フラグの設定
    col1, col2 = st.columns(2)
    
    with col1:
        show_playback_pos = st.checkbox(
            "再生位置線を表示",
            value=player_state['show_playback_position'],
            key="toggle_playback_position"
        )
        player_state['show_playback_position'] = show_playback_pos
    
    with col2:
        show_playback_range = st.checkbox(
            "再生範囲を表示",
            value=player_state['show_playback_range'],
            key="toggle_playback_range"
        )
        player_state['show_playback_range'] = show_playback_range
    
    st.markdown("**🔧 キャッシュ管理**")
    if st.button("🗑️ 範囲音声キャッシュをクリア", key="clear_cache_button"):
        clear_range_audio_cache()
        st.success("キャッシュをクリアしました")
    
    # キャッシュ状況表示
    if 'range_audio_cache' in st.session_state and st.session_state.range_audio_cache:
        st.markdown("**📂 キャッシュ状況**")
        for key, data in st.session_state.range_audio_cache.items():
            file_exists = "✅" if os.path.exists(data['path']) else "❌"
            st.caption(f"{file_exists} {data['label']} ({data['duration']:.1f}秒)")

def handle_play_button(segmentator):
    """再生ボタンの処理（Phase 1対応）"""
    player_state = st.session_state.audio_player_state
    
    if player_state['selection_mode'] == 'time_range':
        # 選択範囲再生
        if player_state['selection_start'] is not None and player_state['selection_end'] is not None:
            player_state['play_start_time'] = player_state['selection_start']
            player_state['play_end_time'] = player_state['selection_end']
        else:
            st.warning("再生範囲が選択されていません。波形をクリックして範囲を選択してください。")
            return
    elif player_state['selection_mode'] == 'full':
        # 全体再生
        player_state['play_start_time'] = 0.0
        player_state['play_end_time'] = segmentator.audio_duration
    else:
        # セグメント再生（今は全体再生として扱う）
        player_state['play_start_time'] = 0.0
        player_state['play_end_time'] = segmentator.audio_duration
    
    # Phase 1: 新しい位置管理機能を使用
    start_position = player_state.get('manual_position', player_state['play_start_time'])
    start_playback_from_position(start_position)
    
    st.success(f"再生開始: {player_state['play_start_time']:.1f}s - {player_state['play_end_time']:.1f}s")

def display_interactive_waveform(segmentator):
    """audix対応: リアルタイム同期波形表示"""
    st.markdown("#### 🎵 高度音声プレーヤー (Audix)")
    st.caption("💡 リアルタイム再生位置追跡 | 範囲選択対応 | 完全同期")
    
    # audixコンポーネントのインポート
    try:
        from streamlit_advanced_audio import audix, WaveSurferOptions, CustomizedRegion
    except ImportError:
        st.error("🚨 streamlit-advanced-audio がインストールされていません")
        st.code("pip install streamlit-advanced-audio")
        # フォールバック：既存のPlotly表示
        display_fallback_waveform(segmentator)
        return
    
    # 波形データ取得
    time_axis, audio_data, sample_rate = segmentator.get_waveform_data()
    player_state = st.session_state.audio_player_state
    
    # audix用の設定
    wavesurfer_options = WaveSurferOptions(
        wave_color="#1f77b4",          # 波形の色
        progress_color="#ff4444",       # 再生位置の色
        height=200,                     # 波形の高さ
        bar_width=1,                   # バーの幅
        bar_gap=0,                     # バーの間隔
        normalize=True                 # 正規化
    )
    
    # 既存セグメントを範囲として表示
    regions = []
    segments = segmentator.get_segments_list()
    segment_colors = ['rgba(255, 127, 14, 0.3)', 'rgba(44, 160, 44, 0.3)', 
                     'rgba(214, 39, 40, 0.3)', 'rgba(148, 103, 189, 0.3)']
    
    for i, segment in enumerate(segments):
        color = segment_colors[i % len(segment_colors)]
        
        regions.append(CustomizedRegion(
            start=segment.start_time,
            end=segment.end_time,
            color=color
        ))
    
    # 再生範囲がある場合は範囲として追加
    if (player_state['play_end_time'] and 
        player_state['play_start_time'] < player_state['play_end_time']):
        regions.append(CustomizedRegion(
            start=player_state['play_start_time'],
            end=player_state['play_end_time'],
            color="rgba(173, 216, 230, 0.5)"
        ))
    
    # 2カラムレイアウト：audixプレーヤー + コントロールパネル  
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # audixプレーヤーを表示
        result = audix(
            audio_data,
            sample_rate=sample_rate,
            wavesurfer_options=wavesurfer_options,
            customized_regions=regions if regions else [],
            key="main_audio_player"
        )
        
        # リアルタイム同期処理
        sync_audio_player_state(result, player_state)
        
        # 選択範囲の処理
        handle_region_selection(result, player_state)
    
    with col2:
        # 右側コントロールパネル
        display_audio_control_panel(segmentator, player_state)

def display_audio_control_panel(segmentator, player_state):
    """右側音声コントロールパネル"""
    st.markdown("### 📋 範囲選択とセグメント追加")
    
    # 数値入力による範囲選択（現在位置ボタン付き）
    col_a, col_b = st.columns(2)
    with col_a:
        # 開始時間設定エリア
        st.markdown("**開始時間**")
        subcol1, subcol2 = st.columns([1, 3])
        with subcol1:
            if st.button("📍", key="set_start_from_current", 
                        help="現在の再生位置を開始時間に設定"):
                set_segment_time_from_current_position('start')
        with subcol2:
            quick_start = st.number_input(
                "開始(秒)", 
                min_value=0.0, 
                max_value=float(segmentator.audio_duration),
                value=float(st.session_state.manual_segment_start_time), 
                step=0.5,
                key=f"quick_start_{st.session_state.input_reset_counter}",
                label_visibility="collapsed"
            )
            # セッション状態を更新
            st.session_state.manual_segment_start_time = quick_start
            
    with col_b:
        # 終了時間設定エリア
        st.markdown("**終了時間**")
        subcol1, subcol2 = st.columns([1, 3])
        with subcol1:
            if st.button("📍", key="set_end_from_current",
                        help="現在の再生位置を終了時間に設定"):
                set_segment_time_from_current_position('end')
        with subcol2:
            quick_end = st.number_input(
                "終了(秒)", 
                min_value=0.0, 
                max_value=float(segmentator.audio_duration),
                value=float(st.session_state.manual_segment_end_time), 
                step=0.5,
                key=f"quick_end_{st.session_state.input_reset_counter}",
                label_visibility="collapsed"
            )
            # セッション状態を更新
            st.session_state.manual_segment_end_time = quick_end
    
    # Phase 3: 再生範囲コピー機能（audix選択範囲も考慮）
    st.markdown("**🔄 範囲連携**")
    if player_state['play_end_time']:
        play_duration = player_state['play_end_time'] - player_state['play_start_time']
        st.info(f"現在の再生範囲: {player_state['play_start_time']:.1f}s - {player_state['play_end_time']:.1f}s ({play_duration:.1f}秒)")
        
        if st.button("📋 再生範囲→選択範囲", key="copy_playback_to_selection"):
            player_state['selection_start'] = player_state['play_start_time']
            player_state['selection_end'] = player_state['play_end_time']
            st.success(f"再生範囲を選択範囲にコピーしました: {player_state['play_start_time']:.1f}s - {player_state['play_end_time']:.1f}s")
            st.session_state.input_reset_counter += 1
            st.rerun()
    
    # ボタンを横並びに配置
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📍 この範囲を選択", key="quick_select"):
            # セッション状態から最新の値を使用
            start_time = st.session_state.manual_segment_start_time
            end_time = st.session_state.manual_segment_end_time
            if end_time > start_time:
                player_state['selection_start'] = start_time
                player_state['selection_end'] = end_time
                st.success(f"範囲選択: {start_time:.1f}s - {end_time:.1f}s")
                st.rerun()
            else:
                st.error("終了時間は開始時間より後にしてください")
    
    with col_btn2:
        # セグメント追加ボタン
        if player_state['selection_start'] is not None and player_state['selection_end'] is not None:
            if st.button("➕ セグメント追加", type="primary", key="add_segment_right_panel"):
                result = st.session_state.manual_segmentator.create_segment(
                    player_state['selection_start'], player_state['selection_end']
                )
                if result.success:
                    st.success(result.message)
                    # 選択をリセット
                    player_state['selection_start'] = None
                    player_state['selection_end'] = None
                    st.session_state.input_reset_counter += 1
                    st.rerun()
                else:
                    st.error(result.message)
        else:
            st.button("➕ セグメント追加", disabled=True, help="先に範囲を選択してください")
    
    # 選択状態の表示
    if player_state['selection_start'] is not None and player_state['selection_end'] is not None:
        duration = player_state['selection_end'] - player_state['selection_start']
        st.success(f"✅ **選択済み**\n\n🕐 {player_state['selection_start']:.1f}s - {player_state['selection_end']:.1f}s\n\n⏱️ 長さ: {duration:.1f}秒")
        
        # クリアボタン
        if st.button("🗑️ 選択をクリア", key="clear_selection_right"):
            player_state['selection_start'] = None
            player_state['selection_end'] = None
            st.session_state.input_reset_counter += 1
            st.success("選択をクリアしました")
            st.rerun()
    else:
        st.info("📝 **上記で範囲を選択してください**")

def sync_audio_player_state(result, player_state):
    """audixとsession_stateの完全同期"""
    if result:
        # リアルタイム位置同期
        current_time = result.get("currentTime", 0.0)
        is_playing = result.get("isPlaying", False)
        
        # session_stateを更新
        player_state['current_position'] = current_time
        player_state['is_playing'] = is_playing
        
        # 再生状態の表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_icon = "🔴" if is_playing else "⚪"
            status_text = "再生中" if is_playing else "停止中"
            st.write(f"{status_icon} **{status_text}**")
        
        with col2:
            st.write(f"📍 **位置**: {current_time:.2f}秒")
        
        with col3:
            # 総時間からの進捗
            total_time = player_state.get('play_end_time') or 30.0  # デフォルト値
            progress = min(current_time / total_time, 1.0) if total_time > 0 else 0.0
            st.progress(progress)

def handle_region_selection(result, player_state):
    """範囲選択の処理"""
    if result and result.get("selectedRegion"):
        region = result["selectedRegion"]
        start_time = region.get("start", 0.0)
        end_time = region.get("end", 0.0)
        
        if start_time < end_time:
            # 選択範囲をsession_stateに反映
            player_state['selection_start'] = start_time
            player_state['selection_end'] = end_time
            
            # 選択情報を表示
            duration = end_time - start_time
            st.success(f"✅ **範囲選択完了**: {start_time:.2f}s - {end_time:.2f}s (長さ: {duration:.2f}秒)")

def display_fallback_waveform(segmentator):
    """audixが利用できない場合のフォールバック表示"""
    st.warning("⚠️ 高度音声プレーヤーが利用できません。基本表示を使用中...")
    
    # 既存のPlotly表示（簡素版）
    time_axis, audio_data, sample_rate = segmentator.get_waveform_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='音声波形',
        line=dict(color='#1f77b4', width=1.5),
        showlegend=False
    ))
    
    # 既存セグメントを表示
    segments = segmentator.get_segments_list()
    segment_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    for i, segment in enumerate(segments):
        color = segment_colors[i % len(segment_colors)]
        speaker_name = segment.assigned_speaker or segment.top_speaker or f"セグメント{segment.id}"
        
        # セグメント領域をハイライト
        fig.add_vrect(
            x0=segment.start_time,
            x1=segment.end_time,
            fillcolor=color,
            opacity=0.25,
            annotation_text=f"#{segment.id}: {speaker_name}",
            annotation_position="top left",
            annotation=dict(
                font=dict(size=10, color='white'),
                bgcolor=color,
                bordercolor='white',
                borderwidth=1
            )
        )
    
    # 選択範囲のハイライト表示（最前面）
    if player_state['selection_start'] is not None and player_state['selection_end'] is not None:
        selection_start = min(player_state['selection_start'], player_state['selection_end'])
        selection_end = max(player_state['selection_start'], player_state['selection_end'])
        duration = selection_end - selection_start
        
        fig.add_vrect(
            x0=selection_start,
            x1=selection_end,
            fillcolor='rgba(255, 255, 0, 0.3)',  # 黄色で半透明
            line=dict(color='gold', width=3),
            annotation_text=f"選択範囲: {duration:.1f}秒",
            annotation_position="top left",
            annotation=dict(
                font=dict(size=12, color='black'),
                bgcolor='gold',
                bordercolor='orange',
                borderwidth=2
            )
        )
        
        # 選択範囲の境界線
        fig.add_vline(
            x=selection_start,
            line=dict(color='orange', width=3, dash='solid'),
            annotation_text=f"開始: {selection_start:.1f}s"
        )
        fig.add_vline(
            x=selection_end,
            line=dict(color='red', width=3, dash='solid'),
            annotation_text=f"終了: {selection_end:.1f}s"
        )
    
    # レイアウト設定
    fig.update_layout(
        title="🎵 音声波形（フォールバック表示）",
        xaxis_title="時間 (秒)",
        yaxis_title="振幅",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)




def display_segment_creation_ui():
    """セグメント作成の操作ガイド"""
    st.subheader("✂️ セグメント作成ガイド")
    
    # 選択状態の確認表示
    player_state = st.session_state.audio_player_state
    if player_state['selection_start'] is not None and player_state['selection_end'] is not None:
        start_time = player_state['selection_start']
        end_time = player_state['selection_end']
        duration = end_time - start_time
        st.success(f"✅ **現在の選択**: {start_time:.1f}s - {end_time:.1f}s (長さ: {duration:.1f}秒)")
        st.info("👆 上記の波形エリア右側の「➕ セグメント追加」ボタンでセグメントを作成できます")
    else:
        st.info("📝 **範囲が未選択**: 上記の波形エリア右側で範囲を選択してください")
    
    st.markdown("""
    ### 📋 使用方法
    
    1. **範囲選択**: 波形表示右側の「範囲選択とセグメント追加」で開始・終了時間を入力
    2. **選択実行**: 「📍 この範囲を選択」をクリック
    3. **セグメント追加**: 「➕ セグメント追加」をクリック
    4. **完了**: セグメントが波形上に色付きで表示されます
    
    ### ⚠️ 注意点
    
    - **最小長さ**: 0.5秒以上
    - **重複禁止**: 既存セグメントとの重複は不可
    - **範囲制限**: 音声範囲内の時間のみ指定可能
    
    ### 💡 ヒント
    
    - 波形を見ながら音声の区切りを確認して範囲を決めてください
    - セグメント追加後は自動的に選択がクリアされます
    - 間違えた場合は「🗑️ 選択をクリア」で選択を取り消せます
    """)

def display_segment_management(segments):
    """セグメント管理表示"""
    st.subheader("📋 Step 3: セグメント一覧")
    
    if not segments:
        st.info("セグメントがありません。上記でセグメントを作成してください。")
        return
    
    # セグメント一覧テーブル
    segment_data = []
    for segment in segments:
        status = "認識済み" if segment.is_recognized else "未認識"
        confidence = segment.confidence_level or "-"
        top_speaker = segment.top_speaker or "-"
        top_score = f"{segment.top_score:.3f}" if segment.top_score else "-"
        
        segment_data.append({
            "ID": segment.id,
            "開始": f"{segment.start_time:.1f}s",
            "終了": f"{segment.end_time:.1f}s",
            "長さ": f"{segment.duration:.1f}s",
            "状態": status,
            "最上位候補": top_speaker,
            "スコア": top_score,
            "信頼度": confidence
        })
    
    df = pd.DataFrame(segment_data)
    st.dataframe(df, use_container_width=True)
    
    # セグメント操作
    st.subheader("🔧 セグメント操作")
    col1, col2 = st.columns(2)
    
    with col1:
        delete_id = st.selectbox(
            "削除するセグメント",
            options=[0] + [s.id for s in segments],
            format_func=lambda x: "選択してください" if x == 0 else f"セグメント {x}"
        )
        
        if st.button("🗑️ 削除") and delete_id > 0:
            result = st.session_state.manual_segmentator.delete_segment(delete_id)
            if result.success:
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)
    
    with col2:
        st.write("**全セグメントクリア**")
        if st.button("🗑️ 全削除", help="すべてのセグメントを削除します"):
            st.session_state.manual_segmentator.segment_manager.clear_segments()
            st.success("全セグメントを削除しました")
            st.rerun()

def display_recognition_execution():
    """認識実行UI"""
    st.subheader("🎯 Step 4: 認識実行")
    
    segments = st.session_state.manual_segmentator.get_segments_list()
    unrecognized_count = sum(1 for s in segments if not s.is_recognized)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**認識対象**: {len(segments)}セグメント (未認識: {unrecognized_count})")
        
        if st.button("🚀 全セグメント認識実行", type="primary", disabled=len(segments)==0):
            with st.spinner("認識処理中..."):
                progress_bar = st.progress(0)
                
                # 各セグメントを個別に処理してプログレスバー更新
                total_segments = len(segments)
                success_count = 0
                
                # バッチ認識実行
                result = st.session_state.manual_segmentator.recognize_all_segments(
                    show_jvs=st.session_state.manual_show_jvs,
                    show_cv=st.session_state.manual_show_cv
                )
                
                progress_bar.progress(100)
                
                if result.success:
                    st.success(result.message)
                    st.rerun()
                else:
                    st.error(result.message)
    
    with col2:
        if unrecognized_count > 0:
            st.info(f"💡 {unrecognized_count}個のセグメントが\n未認識です")
        else:
            st.success("✅ すべて認識済み")

def display_manual_segmentation_results(segments):
    """手動分離結果表示"""
    if not any(s.is_recognized for s in segments):
        return
    
    # 認識結果詳細表示
    st.subheader("🏆 Step 5: 認識結果詳細")
    
    for segment in segments:
        if not segment.is_recognized:
            continue
        
        with st.expander(f"セグメント {segment.id} ({segment.start_time:.1f}s - {segment.end_time:.1f}s)"):
            st.write(f"**長さ**: {segment.duration:.1f}秒")
            st.write(f"**信頼度**: {segment.confidence_level}")
            
            if segment.recognition_results:
                st.write("**Top-5 認識候補:**")
                for result in segment.recognition_results[:5]:
                    rank_icon = "🥇" if result['rank'] == 1 else "🥈" if result['rank'] == 2 else "🥉" if result['rank'] == 3 else f"{result['rank']}."
                    st.write(f"{rank_icon} {result['speaker']}: `{result['score']:.3f}`")
    
    # 話者割り当てと最終結果
    display_speaker_assignment_and_timeline(segments)

def display_speaker_assignment_and_timeline(segments):
    """話者割り当てと最終タイムライン表示"""
    recognized_segments = [s for s in segments if s.is_recognized]
    if not recognized_segments:
        return
    
    st.subheader("🏷️ Step 6: 話者割り当て")
    
    # 登録話者一覧取得
    available_speakers = []
    if st.session_state.recognizer and st.session_state.recognizer.speaker_embeddings:
        all_speakers = list(st.session_state.recognizer.speaker_embeddings.keys())
        
        # カスタム話者
        custom_speakers = [s for s in all_speakers if not (s.startswith('jvs') or s.startswith('cv_') or s.startswith('commonvoice_'))]
        available_speakers.extend(custom_speakers)
        
        # JVS話者（設定により表示）
        if st.session_state.manual_show_jvs:
            jvs_speakers = [s for s in all_speakers if s.startswith('jvs')]
            available_speakers.extend([f"{s} 🗾" for s in jvs_speakers])
        
        # Common Voice話者（設定により表示）
        if st.session_state.manual_show_cv:
            cv_speakers = [s for s in all_speakers if s.startswith(('cv_', 'commonvoice_'))]
            available_speakers.extend([f"{s} 🌐" for s in cv_speakers])
    
    available_speakers.extend(["unknown", "新規話者..."])
    
    # セグメント毎の話者割り当て
    assignment_changed = False
    
    for segment in recognized_segments:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.write(f"**セグメント {segment.id}**")
            st.write(f"({segment.start_time:.1f}s - {segment.end_time:.1f}s)")
        
        with col2:
            current_assignment = segment.assigned_speaker or segment.top_speaker or "unknown"
            
            # 現在の割り当てがリストにない場合は追加
            if current_assignment not in available_speakers and current_assignment:
                available_speakers.insert(-2, current_assignment)
            
            # 話者選択
            selected_speaker = st.selectbox(
                "話者割り当て",
                options=available_speakers,
                index=available_speakers.index(current_assignment) if current_assignment in available_speakers else 0,
                key=f"speaker_assignment_{segment.id}"
            )
            
            # 新規話者の場合
            if selected_speaker == "新規話者...":
                new_speaker = st.text_input(f"新規話者名 (セグメント{segment.id})", key=f"new_speaker_{segment.id}")
                if new_speaker:
                    selected_speaker = new_speaker
            
            # 割り当て更新
            if selected_speaker != segment.assigned_speaker and selected_speaker != "新規話者...":
                segment.assigned_speaker = selected_speaker
                assignment_changed = True
        
        with col3:
            # 予測結果表示
            if segment.top_speaker:
                st.write("**予測結果**")
                st.write(f"{segment.top_speaker}")
                st.write(f"`{segment.top_score:.3f}`")
    
    # 最終タイムライン表示
    if assignment_changed or st.button("🎨 タイムライン生成"):
        display_final_timeline()

def display_final_timeline():
    """最終タイムライン表示（Ganttスタイル）"""
    import plotly.colors as pc
    
    st.subheader("📊 最終結果: 話者別タイムライン")
    
    timeline_data = st.session_state.manual_segmentator.get_timeline_data()
    
    if not timeline_data['speakers']:
        st.info("話者が割り当てられたセグメントがありません")
        return
    
    # 話者一覧とカラーマッピング
    speakers = list(timeline_data['speakers'].keys())
    speakers.sort()  # 一貫した順序
    colors = pc.qualitative.Set2[:len(speakers)]
    speaker_colors = dict(zip(speakers, colors))
    
    # Plotly Ganttスタイルタイムライン作成
    fig = go.Figure()
    
    # 各セグメントをGanttチャートとして追加
    for speaker, data in timeline_data['speakers'].items():
        for segment in data['segments']:
            # 信頼度の表示用テキストを準備
            confidence_text = f"{segment['confidence']:.3f}" if segment['confidence'] is not None else "N/A"
            
            # ホバー情報
            hover_text = (
                f"話者: {speaker}<br>"
                f"時間: {segment['start']:.1f}s - {segment['end']:.1f}s<br>"
                f"時間長: {segment['duration']:.1f}s<br>"
                f"信頼度: {confidence_text}"
            )
            
            fig.add_trace(go.Bar(
                x=[segment['duration']],
                y=[speaker],
                base=segment['start'],
                orientation='h',
                name=speaker,
                marker_color=speaker_colors[speaker],
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False
            ))
    
    # レイアウト設定
    fig.update_layout(
        title="👥 話者別発話タイムライン",
        xaxis_title="時間（秒）",
        yaxis_title="話者",
        height=max(300, len(speakers) * 60),
        showlegend=False,
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 統計情報
    display_manual_statistics(timeline_data)
    
    # エクスポート機能
    display_export_options()

def display_manual_statistics(timeline_data):
    """統計情報表示"""
    st.subheader("📈 統計情報")
    
    stats_data = []
    total_speech_time = 0
    
    for speaker, data in timeline_data['speakers'].items():
        total_time = data['total_time']
        segment_count = data['segment_count']
        avg_confidence = np.mean([s['confidence'] for s in data['segments'] if s['confidence'] is not None])
        speech_ratio = (total_time / timeline_data['total_duration']) * 100
        
        stats_data.append({
            "話者名": speaker,
            "セグメント数": segment_count,
            "総発話時間": f"{total_time:.1f}秒",
            "平均信頼度": f"{avg_confidence:.3f}" if not np.isnan(avg_confidence) else "-",
            "発話割合": f"{speech_ratio:.1f}%"
        })
        
        total_speech_time += total_time
    
    # 統計テーブル
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
    
    # サマリー情報
    silence_time = timeline_data['total_duration'] - total_speech_time
    silence_ratio = (silence_time / timeline_data['total_duration']) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総音声長", f"{timeline_data['total_duration']:.1f}秒")
    with col2:
        st.metric("総発話時間", f"{total_speech_time:.1f}秒")
    with col3:
        st.metric("無音時間", f"{silence_time:.1f}秒")
    with col4:
        st.metric("無音割合", f"{silence_ratio:.1f}%")

def display_export_options():
    """エクスポートオプション"""
    st.subheader("💾 結果エクスポート")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 CSV形式"):
            success, message = st.session_state.manual_segmentator.export_to_csv("manual_segmentation_results.csv")
            if success:
                st.success(message)
                with open("manual_segmentation_results.csv", "rb") as f:
                    st.download_button(
                        label="📁 CSVダウンロード",
                        data=f.read(),
                        file_name="manual_segmentation_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error(message)
    
    with col2:
        if st.button("📄 JSON形式"):
            success, message = st.session_state.manual_segmentator.export_to_json("manual_segmentation_results.json")
            if success:
                st.success(message)
                with open("manual_segmentation_results.json", "rb") as f:
                    st.download_button(
                        label="📁 JSONダウンロード",
                        data=f.read(),
                        file_name="manual_segmentation_results.json",
                        mime="application/json"
                    )
            else:
                st.error(message)
    
    with col3:
        if st.button("📄 SRT字幕"):
            success, message = st.session_state.manual_segmentator.export_to_srt("manual_segmentation_results.srt")
            if success:
                st.success(message)
                with open("manual_segmentation_results.srt", "rb") as f:
                    st.download_button(
                        label="📁 SRTダウンロード",
                        data=f.read(),
                        file_name="manual_segmentation_results.srt",
                        mime="text/plain"
                    )
            else:
                st.error(message)
    
    with col4:
        st.write("**🖼️ PNG画像**")
        st.caption("タイムライン画像として保存")
        st.info("開発予定")


if __name__ == "__main__":
    main()