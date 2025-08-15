"""
Common Voice 日本語データセットから背景埋め込みを作成するスクリプト

Hugging Face DatasetsのストリーミングAPIを使用して、
全量ダウンロードなしでCommon Voice日本語データセットから
背景埋め込みを作成します。

Common VoiceはCC0ライセンスのため、作成した埋め込みは
自由に再配布・同梱可能です。

使用法:
python prep_common_voice_embeddings.py [--max-samples 5000]
"""

import numpy as np
import torch
import librosa
import logging
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
import argparse
from pathlib import Path
import time

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

def extract_embedding(model, audio_array, sample_rate, device, target_sr=16000):
    """音声配列から埋め込みベクトルを抽出"""
    try:
        # サンプリングレート変換
        if sample_rate != target_sr:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=target_sr
            )
        
        # 音声の長さをチェック（短すぎる音声は除外）
        if len(audio_array) < target_sr * 1.0:  # 1秒未満は除外
            return None
        
        # 長すぎる音声は切り詰め（処理時間短縮のため）
        max_length = target_sr * 30  # 30秒
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        
        with torch.no_grad():
            # バッチ次元を追加してデバイスに送る
            wav_tensor = torch.from_numpy(audio_array.astype('float32')).unsqueeze(0).to(device)
            embedding = model.encode_batch(wav_tensor)
        
        # ベクトルを正規化
        vector = embedding.squeeze(0).squeeze(0).cpu().numpy()
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-10)
        
        return normalized_vector
    except Exception as e:
        logger.warning(f"Failed to extract embedding: {e}")
        return None

def collect_common_voice_embeddings(model, device, max_samples=5000, max_duration=12.0):
    """Common Voice日本語データセットから埋め込みを収集（ストリーミング）"""
    
    logger.info("Loading Common Voice Japanese dataset (streaming)...")
    
    # ストリーミングモードでデータセットを読み込み
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0", 
            "ja", 
            split="train", 
            streaming=True
        )
    except Exception as e:
        logger.error(f"Failed to load Common Voice 17.0: {e}")
        logger.info("Trying older version (Common Voice 16.1)...")
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_16_1", 
                "ja", 
                split="train", 
                streaming=True
            )
            logger.info("Successfully loaded Common Voice 16.1")
        except Exception as e2:
            logger.error(f"Failed to load Common Voice 16.1: {e2}")
            logger.info("Trying Common Voice 13.0...")
            try:
                dataset = load_dataset(
                    "mozilla-foundation/common_voice_13_0", 
                    "ja", 
                    split="train", 
                    streaming=True
                )
                logger.info("Successfully loaded Common Voice 13.0")
            except Exception as e3:
                raise RuntimeError(f"Failed to load any Common Voice version: {e}, {e2}, {e3}")
    
    embeddings = []
    processed_count = 0
    skipped_count = 0
    
    logger.info(f"Processing up to {max_samples} samples...")
    start_time = time.time()
    
    for i, example in enumerate(dataset):
        if len(embeddings) >= max_samples:
            break
        
        try:
            # 音声データを取得
            audio_data = example["audio"]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]
            
            if audio_array is None or len(audio_array) == 0:
                skipped_count += 1
                continue
            
            # 音声の長さをチェック
            duration = len(audio_array) / sample_rate
            if duration > max_duration or duration < 1.0:
                skipped_count += 1
                continue
            
            # 埋め込みを抽出
            embedding = extract_embedding(model, audio_array, sample_rate, device)
            if embedding is not None:
                embeddings.append(embedding)
                processed_count += 1
                
                # 進捗表示
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    logger.info(f"  Processed: {processed_count}, Skipped: {skipped_count}, Rate: {rate:.1f} samples/sec")
            else:
                skipped_count += 1
                
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            skipped_count += 1
            continue
    
    if not embeddings:
        raise ValueError("No embeddings were extracted")
    
    elapsed = time.time() - start_time
    logger.info(f"Collection completed in {elapsed:.1f} seconds")
    logger.info(f"Total processed: {processed_count}, Total skipped: {skipped_count}")
    
    return np.stack(embeddings)

def main():
    parser = argparse.ArgumentParser(description='Create Common Voice Japanese background embeddings')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum number of samples to process (default: 5000)')
    parser.add_argument('--max-duration', type=float, default=12.0,
                        help='Maximum audio duration in seconds (default: 12.0)')
    parser.add_argument('--output', default='background_common_voice_ja_ecapa.npz',
                        help='Output file name (default: background_common_voice_ja_ecapa.npz)')
    
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
    
    # Common Voice埋め込みを収集
    logger.info("Collecting embeddings from Common Voice Japanese...")
    embeddings = collect_common_voice_embeddings(
        model, device, args.max_samples, args.max_duration
    )
    
    # 結果を保存
    logger.info(f"Saving {len(embeddings)} embeddings to {args.output}")
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        model_info="speechbrain/spkrec-ecapa-voxceleb",
        dataset_info="mozilla-foundation/common_voice_17_0 ja"
    )
    
    logger.info(f"Saved embeddings shape: {embeddings.shape}")
    logger.info("Done!")

if __name__ == "__main__":
    main()