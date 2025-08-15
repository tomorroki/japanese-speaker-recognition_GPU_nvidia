#!/usr/bin/env python3
"""
背景埋め込み作成用の統合スクリプト

JVSとCommon Voice日本語データセットの両方から
背景埋め込みを作成するための便利なスクリプト
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """コマンドを実行"""
    logger.info(f"実行中: {description}")
    logger.info(f"コマンド: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"成功: {description}")
        if result.stdout:
            logger.info(f"出力: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"失敗: {description}")
        logger.error(f"エラー: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create background embeddings from JVS and Common Voice')
    
    # JVS関連
    parser.add_argument('--jvs-path', help='Path to JVS corpus directory')
    parser.add_argument('--jvs-per-speaker', type=int, default=50, 
                        help='Number of files per JVS speaker (default: 50)')
    parser.add_argument('--skip-jvs', action='store_true',
                        help='Skip JVS embedding creation')
    
    # Common Voice関連
    parser.add_argument('--cv-max-samples', type=int, default=5000,
                        help='Maximum samples from Common Voice (default: 5000)')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip Common Voice embedding creation')
    
    # 出力設定
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for embedding files (default: current directory)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_tasks = 0
    
    # JVS埋め込み作成
    if not args.skip_jvs:
        total_tasks += 1
        if args.jvs_path:
            jvs_output = output_dir / "background_jvs_ecapa.npz"
            cmd = [
                sys.executable, "prep_jvs_embeddings.py",
                args.jvs_path,
                "--per-speaker", str(args.jvs_per_speaker),
                "--output", str(jvs_output)
            ]
            
            if run_command(cmd, "JVS埋め込み作成"):
                success_count += 1
                logger.info(f"JVS埋め込みファイル: {jvs_output}")
        else:
            logger.warning("JVSパスが指定されていません。--jvs-pathを指定してください")
    
    # Common Voice埋め込み作成
    if not args.skip_cv:
        total_tasks += 1
        cv_output = output_dir / "background_common_voice_ja_ecapa.npz"
        cmd = [
            sys.executable, "prep_common_voice_embeddings.py",
            "--max-samples", str(args.cv_max_samples),
            "--output", str(cv_output)
        ]
        
        if run_command(cmd, "Common Voice埋め込み作成"):
            success_count += 1
            logger.info(f"Common Voice埋め込みファイル: {cv_output}")
    
    # 結果サマリー
    logger.info(f"\n=== 作成完了 ===")
    logger.info(f"成功: {success_count}/{total_tasks} タスク")
    
    if success_count > 0:
        logger.info(f"\n作成されたファイル:")
        if not args.skip_jvs and args.jvs_path:
            jvs_file = output_dir / "background_jvs_ecapa.npz"
            if jvs_file.exists():
                logger.info(f"  - {jvs_file}")
        
        if not args.skip_cv:
            cv_file = output_dir / "background_common_voice_ja_ecapa.npz"
            if cv_file.exists():
                logger.info(f"  - {cv_file}")
        
        logger.info(f"\nこれらのファイルをアプリケーションディレクトリまたは")
        logger.info(f"background_embeddings/フォルダに配置してください。")
    
    if success_count == total_tasks:
        logger.info("\n全て正常に完了しました！")
        return 0
    else:
        logger.error(f"\n一部のタスクが失敗しました。上記のエラーを確認してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main())