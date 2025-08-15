"""
Common Voice用のダミー埋め込みファイルを作成

認証や大きなダウンロードなしで、とりあえずシステムを動作させるためのファイル
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_embeddings(output_file="background_common_voice_ja_ecapa.npz", num_samples=1000, embedding_dim=192):
    """ダミー埋め込みを作成"""
    
    logger.info(f"Creating dummy Common Voice embeddings: {num_samples} samples")
    
    # ランダムなダミー埋め込みを生成（正規化済み）
    embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    
    # L2正規化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    # ファイル保存
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        model_info="dummy_ecapa_for_testing",
        dataset_info="dummy_common_voice_ja"
    )
    
    logger.info(f"Saved dummy embeddings: {embeddings.shape} to {output_file}")
    logger.info("NOTE: これはダミーデータです。実際のCommon Voiceデータを使用するには認証が必要です")

if __name__ == "__main__":
    create_dummy_embeddings()