"""
日本語話者認識システム - Gradio UI
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from enhanced_speaker_recognition import JapaneseSpeakerRecognizer, RecognitionResult

# グローバル変数でrecognizerを管理
recognizer = None

def initialize_system():
    """システムの初期化"""
    global recognizer
    
    try:
        # Recognizer初期化
        recognizer = JapaneseSpeakerRecognizer()
        
        # モデル読み込み
        if not recognizer.initialize_model():
            return "❌ モデルの初期化に失敗しました"
        
        # 話者データベース構築
        enrolled_count = recognizer.build_speaker_database()
        
        # 背景モデル構築
        recognizer.build_background_model()
        
        return f"✅ システム初期化完了！{enrolled_count}名の話者を登録しました"
        
    except Exception as e:
        return f"❌ 初期化エラー: {str(e)}"

def recognize_speaker_from_audio(
    audio_file, 
    show_jvs: bool = True, 
    show_cv: bool = False
) -> Tuple[str, str, str, str, str]:
    """
    音声ファイルから話者を識別
    
    Args:
        audio_file: アップロードされた音声ファイル
        show_jvs: JVS話者を結果に表示するか
        show_cv: Common Voice話者を結果に表示するか
        
    Returns:
        結果のタプル (メイン結果, 詳細情報, トップ10テーブル, グラフHTML, ステータス)
    """
    global recognizer
    
    if recognizer is None:
        return "❌ システムが初期化されていません", "", "", "", "error"
    
    if audio_file is None:
        return "📁 音声ファイルをアップロードしてください", "", "", "", "info"
    
    try:
        # 話者識別実行
        result = recognizer.recognize_speaker(audio_file)
        
        if result is None:
            return "❌ 話者識別に失敗しました", "", "", "", "error"
        
        # 表示設定に基づいてスコアをフィルタリング
        if result.all_scores:
            filtered_scores = recognizer.filter_scores_for_display(
                result.all_scores, show_jvs, show_cv
            )
        else:
            filtered_scores = {}
        
        # フィルタリング後の最上位話者を決定
        display_speaker = result.speaker_id
        display_confidence = result.confidence
        display_raw_score = result.raw_score
        
        if filtered_scores:
            filtered_best_speaker = max(filtered_scores, key=filtered_scores.get)
            filtered_best_score = filtered_scores[filtered_best_speaker]
            
            # オリジナルの最上位話者が除外される場合
            if filtered_best_speaker != result.speaker_id:
                is_original_jvs = recognizer.dataset_manager.is_jvs_speaker(result.speaker_id)
                is_original_cv = recognizer.dataset_manager.is_common_voice_speaker(result.speaker_id)
                
                if (is_original_jvs and not show_jvs) or (is_original_cv and not show_cv):
                    display_speaker = filtered_best_speaker
                    display_raw_score = filtered_best_score
                    display_confidence = filtered_best_score
        
        # メイン結果の生成
        threshold = recognizer.threshold
        confidence_emoji = "🟢" if display_confidence > 0.5 else "🟡" if display_confidence > 0.25 else "🔴"
        threshold_status = "✅ 閾値を上回りました" if display_confidence > threshold else "⚠️ 閾値を下回りました"
        
        main_result = f"""
## 🎯 識別結果

**識別された話者**: `{display_speaker}`  
**信頼度**: {confidence_emoji} `{display_confidence:.3f}`  
**生スコア**: `{display_raw_score:.3f}`  
**閾値判定**: {threshold_status} (閾値: {threshold:.3f})
        """
        
        # 詳細情報
        detail_info = f"""
### 📊 詳細情報

- **元の最上位話者**: {result.speaker_id}
- **元の信頼度**: {result.confidence:.3f}
- **正規化スコア**: {f"{result.normalized_score:.3f}" if result.normalized_score is not None else 'N/A'}
- **フィルタリング適用**: {'あり' if show_jvs != True or show_cv != False else 'なし'}
        """
        
        # トップ10テーブルの生成
        if filtered_scores:
            # filter_scores_for_displayは既にソート済みなので、そのまま使用
            # DataFrameを作成
            df_data = []
            for i, (speaker_id, score) in enumerate(filtered_scores.items(), 1):
                df_data.append({
                    "順位": i,
                    "話者ID": speaker_id,
                    "類似度スコア": f"{score:.3f}",
                    "タイプ": get_speaker_type(speaker_id, recognizer)
                })
            
            df = pd.DataFrame(df_data)
            top10_table = df.to_html(index=False, escape=False, classes="gradio-table")
        else:
            top10_table = "<p>表示する結果がありません</p>"
        
        # グラフの生成
        chart_html = create_score_chart(filtered_scores, display_speaker)
        
        return main_result, detail_info, top10_table, chart_html, "success"
        
    except Exception as e:
        return f"❌ 識別エラー: {str(e)}", "", "", "", "error"

def get_speaker_type(speaker_id: str, recognizer) -> str:
    """話者のタイプを取得"""
    if recognizer.dataset_manager.is_jvs_speaker(speaker_id):
        return "🗾 JVS"
    elif recognizer.dataset_manager.is_common_voice_speaker(speaker_id):
        return "🌐 CV"
    else:
        return "👤 カスタム"

def create_score_chart(scores: Dict[str, float], best_speaker: str) -> str:
    """スコアチャートのHTML生成"""
    if not scores:
        return "<p>グラフを生成するデータがありません</p>"
    
    # データ準備（filter_scores_for_displayから既にソート済み）
    speakers = list(scores.keys())
    score_values = list(scores.values())
    colors = ['#ff6b6b' if speaker == best_speaker else '#74c0fc' for speaker in speakers]
    
    # Plotlyグラフ作成
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
        title="話者別類似度スコア (Top 10)",
        xaxis_title="類似度スコア",
        yaxis_title="話者ID",
        height=max(300, len(speakers) * 40),
        showlegend=False,
        template="plotly_white",
        yaxis=dict(autorange="reversed")  # 上位から下位の順で表示
    )
    
    return fig.to_html(include_plotlyjs='cdn')

def get_system_info():
    """システム情報を取得"""
    global recognizer
    
    if recognizer is None:
        return "システムが初期化されていません"
    
    try:
        info = recognizer.get_system_info()
        
        system_info = f"""
### 🔧 システム情報

**登録話者数**: {info["enrolled_speakers"]}  
**サンプリングレート**: {info["sample_rate"]} Hz  
**背景サンプル数**: {info["background_samples"]}  
**しきい値**: {info["threshold"]:.3f}  
**デバイス**: {info["device"]}  
**スコア正規化**: {'有効' if info["score_normalization"] else '無効'}  
**モデル**: {info["model_name"]}
        """
        
        return system_info
        
    except Exception as e:
        return f"システム情報取得エラー: {str(e)}"

# Gradio インターフェースの構築
with gr.Blocks(
    title="日本語話者認識システム",
    theme=gr.themes.Soft(),
    css="""
    .gradio-table { margin: 10px 0; border-collapse: collapse; width: 100%; }
    .gradio-table th, .gradio-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    .gradio-table th { background-color: #f2f2f2; }
    """
) as demo:
    
    gr.Markdown("# 🎤 日本語話者認識システム")
    gr.Markdown("### SpeechBrain + ECAPA-TDNNによる高精度話者識別")
    
    with gr.Tab("🎤 話者識別"):
        with gr.Row():
            with gr.Column():
                # システム初期化
                init_btn = gr.Button("🚀 システム初期化", variant="primary")
                init_status = gr.Textbox(label="初期化状況", interactive=False)
                
                gr.Markdown("---")
                
                # 音声アップロード
                audio_input = gr.Audio(
                    label="音声ファイルをアップロード",
                    type="filepath",
                    format="wav"
                )
                
                # 表示設定
                gr.Markdown("### 🎛️ 表示設定")
                show_jvs = gr.Checkbox(
                    label="🗾 JVS話者を結果に表示",
                    value=True,
                    info="認識結果のトップ10にJVS話者を含めるかどうか"
                )
                show_cv = gr.Checkbox(
                    label="🌐 Common Voice話者を結果に表示",
                    value=False,
                    info="認識結果のトップ10にCommon Voice話者を含めるかどうか"
                )
                
                # 識別実行
                recognize_btn = gr.Button("🔍 話者識別開始", variant="primary")
        
        with gr.Column():
            # 結果表示
            main_result = gr.Markdown(label="識別結果")
            detail_info = gr.Markdown(label="詳細情報")
            
    with gr.Tab("📊 トップ10ランキング"):
        top10_table = gr.HTML(label="トップ10話者スコア")
        score_chart = gr.HTML(label="スコアグラフ")
    
    with gr.Tab("ℹ️ システム情報"):
        system_info_btn = gr.Button("システム情報を更新", variant="secondary")
        system_info_display = gr.Markdown(label="システム情報")
    
    # イベントハンドラーの設定
    init_btn.click(
        fn=initialize_system,
        outputs=init_status
    )
    
    recognize_btn.click(
        fn=recognize_speaker_from_audio,
        inputs=[audio_input, show_jvs, show_cv],
        outputs=[main_result, detail_info, top10_table, score_chart]
    )
    
    system_info_btn.click(
        fn=get_system_info,
        outputs=system_info_display
    )

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7865,
        share=False,
        show_error=True,
        inbrowser=True
    )