"""
日本語話者認識システム - Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from enhanced_speaker_recognition import JapaneseSpeakerRecognizer, RecognitionResult
from dataset_manager import DatasetManager

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎤 単一話者識別", 
        "🎭 複数話者分析", 
        "👥 話者管理", 
        "📊 統計情報"
    ])
    
    with tab1:
        display_recognition_tab()
    
    with tab2:
        display_diarization_tab()
    
    with tab3:
        display_speaker_management_tab()
    
    with tab4:
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
    
    # 初期化確認
    if 'multi_recognizer' not in st.session_state:
        st.session_state.multi_recognizer = None
        st.session_state.diarization_initialized = False
    
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
        col1, col2 = st.columns(2)
        with col1:
            min_speakers = st.number_input("最小話者数", 1, 10, 1, help="音声に含まれる最小話者数")
        with col2:
            max_speakers = st.number_input("最大話者数", 1, 10, 5, help="音声に含まれる最大話者数")
        
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
            perform_multi_speaker_analysis(uploaded_file, min_speakers, max_speakers)

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

def perform_multi_speaker_analysis(uploaded_file, min_speakers, max_speakers):
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
            display_multi_speaker_result(result)
            
        except Exception as e:
            st.error(f"❌ 分析エラー: {e}")
        finally:
            # 一時ファイル削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def display_multi_speaker_result(result):
    """複数話者分析結果表示"""
    st.subheader("🎯 分析結果")
    
    # サマリー
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("検出話者数", result.total_speakers)
    with col2:
        st.metric("総時間", f"{result.total_duration:.1f}秒")
    with col3:
        st.metric("セグメント数", len(result.segments))
    
    # セグメント詳細
    if result.segments:
        st.subheader("📋 時系列セグメント")
        
        for i, segment in enumerate(result.segments):
            # 認識成功のセグメントは緑、失敗は赤で表示
            if segment['recognized_speaker'] != "未認識":
                status_color = "🟢"
                confidence_text = f" (信頼度: {segment['confidence']:.3f})"
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
                        st.write(f"**信頼度**: {segment['confidence']:.3f}")
        
        # 話者別サマリー
        st.subheader("👥 話者別サマリー")
        speaker_summary = {}
        for segment in result.segments:
            speaker = segment['recognized_speaker']
            if speaker not in speaker_summary:
                speaker_summary[speaker] = {
                    'segments': 0,
                    'total_time': 0.0,
                    'avg_confidence': 0.0
                }
            speaker_summary[speaker]['segments'] += 1
            speaker_summary[speaker]['total_time'] += segment['duration']
            if segment['confidence'] > 0:
                speaker_summary[speaker]['avg_confidence'] += segment['confidence']
        
        # 平均信頼度計算
        for speaker in speaker_summary:
            if speaker_summary[speaker]['segments'] > 0:
                speaker_summary[speaker]['avg_confidence'] /= speaker_summary[speaker]['segments']
        
        # 表形式で表示
        summary_data = []
        for speaker, data in speaker_summary.items():
            summary_data.append({
                '話者': speaker,
                'セグメント数': data['segments'],
                '合計時間': f"{data['total_time']:.1f}秒",
                '平均信頼度': f"{data['avg_confidence']:.3f}" if data['avg_confidence'] > 0 else "N/A"
            })
        
        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("⚠️ セグメントが検出されませんでした。音声ファイルや設定を確認してください。")

if __name__ == "__main__":
    main()