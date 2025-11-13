#!/usr/bin/env python3
"""
預覽批次硬體比較腳本執行計劃
================================

這個腳本使用 dry-run 模式預覽所有要執行的命令，
不會實際執行任何實驗。

使用方式：
    # 預覽 G1~G3
    python preview_batch_hardware.py --start 1 --end 3
    
    # 預覽所有圖檔（G1~G81）
    python preview_batch_hardware.py --start 1 --end 81
"""

import subprocess
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="預覽批次硬體比較腳本執行計劃")
    parser.add_argument('--start', type=int, default=1, help="起始圖檔編號 (預設: 1)")
    parser.add_argument('--end', type=int, default=3, help="結束圖檔編號 (預設: 3)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("="*80)
    print("預覽批次硬體比較腳本執行計劃")
    print("="*80)
    print(f"圖檔範圍：G{args.start}~G{args.end}")
    print(f"參數組合：4 種")
    print(f"總實驗數：{(args.end - args.start + 1) * 4}")
    print("="*80)
    print()
    
    # 執行 dry-run
    cmd = [
        sys.executable,
        'batch_hardware_comparison.py',
        '--graph_dir', './graph',
        '--start_graph', str(args.start),
        '--end_graph', str(args.end),
        '--dry_run'
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("預覽完成")
        print("="*80)
        print("\n如果命令看起來正確，可以執行以下命令開始實驗：")
        print(f"  python batch_hardware_comparison.py --start_graph {args.start} --end_graph {args.end}")
        print("="*80)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("✗ 預覽失敗")
        print("="*80)
        print(f"錯誤訊息: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n預覽被中斷")
        sys.exit(1)

if __name__ == "__main__":
    main()

