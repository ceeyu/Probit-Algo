#!/usr/bin/env python3
"""
快速測試批次硬體比較腳本
===========================

這個腳本用於快速測試 batch_hardware_comparison.py 是否正常運作
測試範圍：G1~G3，只測試兩種參數組合

使用方式：
    python test_batch_hardware.py
"""

import subprocess
import sys

def main():
    print("="*80)
    print("快速測試批次硬體比較腳本")
    print("="*80)
    print("測試範圍：G1~G22")
    print("測試參數組合：4 種（完整測試）")
    print("預計時間：約 5-10 分鐘（取決於硬體）")
    print("="*80)
    print()
    
    # 詢問使用者是否要執行
    response = input("是否開始測試？(y/n): ").strip().lower()
    if response != 'y':
        print("取消測試")
        sys.exit(0)
    
    print("\n開始測試...\n")
    
    # 執行批次腳本（只測試 G1~G22）
    cmd = [
        sys.executable,
        'batch_hardware_comparison.py',
        '--graph_dir', './graph',
        '--start_graph', '1',
        '--end_graph', '22'
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✓ 測試完成！")
        print("="*80)
        print("\n請檢查以下資料夾中的結果：")
        print("  - hardware_comparison_results/trial100_steps1000/")
        print("  - hardware_comparison_results/trial100_steps10000/")
        print("  - hardware_comparison_results/trial1000_steps100/")
        print("  - hardware_comparison_results/trial1000_steps10000/")
        print("\n每個資料夾應該包含 G1、G22 的結果檔案（CSV、PNG、XLSX）")
        print("="*80)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("✗ 測試失敗")
        print("="*80)
        print(f"錯誤訊息: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n測試被中斷")
        sys.exit(1)

if __name__ == "__main__":
    main()

