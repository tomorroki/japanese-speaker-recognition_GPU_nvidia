"""
JVS データセットから背景埋め込みを作成するスクリプト

JVSデータセットをローカルでダウンロード・解凍後に実行し、
話者の埋め込みベクトルを事前計算して保存します。
音声ファイルは再配布禁止のため、埋め込みのみをアプリに同梱します。

使用法:
python prep_jvs_embeddings.py /path/to/JVS_corpus
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier
import argparse
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device():
    """デバイス設定"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA device")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    return device

def load_audio_16k(path, target_sr=16000):
    """音声を16kHzで読み込み"""
    try:
        wav, orig_sr = sf.read(path)
        
        # ステレオからモノラルに変換
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # サンプリングレート変換
        if orig_sr != target_sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
        
        return wav.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None

def extract_embedding(model, wav, device):
    """音声から埋め込みベクトルを抽出"""
    try:
        with torch.no_grad():
            # バッチ次元を追加してデバイスに送る
            wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
            embedding = model.encode_batch(wav_tensor)
        
        # ベクトルを正規化
        vector = embedding.squeeze(0).squeeze(0).cpu().numpy()
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-10)
        
        return normalized_vector
    except Exception as e:
        logger.warning(f"Failed to extract embedding: {e}")
        return None

def collect_jvs_embeddings(jvs_root, model, device, per_speaker=50):
    """JVSデータセットから埋め込みを収集"""
    jvs_path = Path(jvs_root)
    
    # JVS話者ディレクトリを検索
    speaker_dirs = sorted([p for p in jvs_path.glob("jvs*") if p.is_dir()])
    
    if not speaker_dirs:
        raise ValueError(f"JVS speaker directories not found in {jvs_root}")
    
    embeddings = []
    speaker_ids = []
    
    logger.info(f"Found {len(speaker_dirs)} JVS speakers")
    
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        logger.info(f"Processing speaker: {speaker_id}")
        
        # 音声ファイルを収集
        wav_files = []
        for subdirs in ["parallel100", "nonpara30"]:
            wav_pattern = speaker_dir / subdirs / "wav24kHz16bit" / "*.wav"
            wav_files.extend(sorted(glob.glob(str(wav_pattern))))
        
        if not wav_files:
            logger.warning(f"No audio files found for {speaker_id}")
            continue
        
        # 指定した数の音声ファイルを処理
        processed_count = 0
        for wav_path in wav_files[:per_speaker]:
            wav = load_audio_16k(wav_path)
            if wav is None or len(wav) < 16000:  # 1秒未満は除外
                continue
            
            embedding = extract_embedding(model, wav, device)
            if embedding is not None:
                embeddings.append(embedding)
                speaker_ids.append(speaker_id)
                processed_count += 1
        
        logger.info(f"  Processed {processed_count} files for {speaker_id}")
    
    if not embeddings:
        raise ValueError("No embeddings were extracted")
    
    return np.stack(embeddings), np.array(speaker_ids)

def main():
    parser = argparse.ArgumentParser(description='Create JVS background embeddings')
    parser.add_argument('jvs_path', help='Path to JVS corpus directory')
    parser.add_argument('--per-speaker', type=int, default=50, 
                        help='Number of files per speaker (default: 50)')
    parser.add_argument('--output', default='background_jvs_ecapa.npz',
                        help='Output file name (default: background_jvs_ecapa.npz)')
    
    args = parser.parse_args()
    
    # デバイス設定
    device = setup_device()
    
    # ECAPAモデルを読み込み
    logger.info("Loading ECAPA-TDNN model...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )
    logger.info("Model loaded successfully")
    
    # JVS埋め込みを収集
    logger.info(f"Collecting embeddings from {args.jvs_path}")
    embeddings, speaker_ids = collect_jvs_embeddings(
        args.jvs_path, model, device, args.per_speaker
    )
    
    # 結果を保存
    logger.info(f"Saving {len(embeddings)} embeddings to {args.output}")
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        speaker_ids=speaker_ids,
        model_info="speechbrain/spkrec-ecapa-voxceleb"
    )
    
    logger.info(f"Saved embeddings shape: {embeddings.shape}")
    logger.info(f"Unique speakers: {len(np.unique(speaker_ids))}")
    logger.info("Done!")

if __name__ == "__main__":
    main()