#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config 4 vs Config 5 快速測試腳本
測試2-3個圖檔以驗證完整流程
"""

import os
import glob
import subprocess
import time
import argparse

def get_small_graphs():
    """獲取較小的圖檔進行快速測試"""
    graph_dir = "./graph"
    all_graphs = sorted(glob.glob(os.path.join(graph_dir, "G*.txt")))
    
    # 選擇檔案大小較小的圖進行測試
    small_graphs = []
    for graph in all_graphs:
        size_mb = os.path.getsize(graph) / (1024 * 1024)
        graph_name = os.path.basename(graph)
        print(f"{graph_name}: {size_mb:.1f}MB")
        if size_mb < 0.3:  # 小於300KB的圖檔
            small_graphs.append(graph)
    
    return small_graphs[:2]  # 只取前2個小圖

def run_single_test(graph_file, config, cycle=200, trial=3):
    """執行單個快速測試"""
    graph_name = os.path.basename(graph_file).replace('.txt', '')
    print(f"\n{'='*50}")
    print(f"快速測試 {graph_name} - Config {config}")
    print(f"參數: cycle={cycle}, trial={trial}")
    print(f"{'='*50}")
    
    if config == 4:
        cmd = [
            'python', 'gpu_MAXCUT_var0708.py',
            '--config', '4',
            '--file_path', graph_file,
            '--cycle', str(cycle),
            '--trial', str(trial),
            '--tau', '8',
            '--res', '2',
            '--l_scale', '0.1',
            '--d_scale', '0.1',
            '--n_scale', '0.1'
        ]
    elif config == 5:
        cmd = [
            'python', 'gpu_MAXCUT_var0708.py',
            '--config', '5',
            '--file_path', graph_file,
            '--cycle', str(cycle),
            '--trial', str(trial),
            '--tau', '8',
            '--res', '2',
            '--l_scale', '0.1',
            '--d_scale', '0.1',
            '--n_scale', '0.1',
            '--stall_threshold', '100',  # 降低閾值以配合較少的cycle
            '--eta_alpha', '0.0',
            '--eta_beta', '0.001',      # 稍微增加以配合較短的測試
            '--target_drop', '10',      # 降低目標以配合較短的測試
            '--window_size', '50'       # 減小視窗大小
        ]
    
    start_time = time.time()
    try:
        print("執行指令:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Config {config} 執行成功 ({end_time - start_time:.1f}秒)")
            
            # 提取關鍵結果
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in [
                    'Average cut:', 'Maximum cut:', 'Average energy:', 
                    'Average annealing time:', 'Total time:', 'GPU Adaptive'
                ]):
                    print(f"  {line}")
            return True
        else:
            print(f"❌ Config {config} 執行失敗")
            print("錯誤輸出:", result.stderr[-500:])  # 只顯示最後500字符
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Config {config} 執行超時 (3分鐘)")
        return False
    except Exception as e:
        print(f"❌ Config {config} 執行錯誤: {e}")
        return False

def quick_comparison_test():
    """執行快速比較測試"""
    print("🧪 Config 4 vs Config 5 快速測試")
    print("="*60)
    
    # 獲取小圖進行測試
    print("選擇測試圖檔...")
    small_graphs = get_small_graphs()
    
    if not small_graphs:
        print("❌ 未找到適合的小圖檔")
        return
    
    print(f"\n選定測試圖檔: {[os.path.basename(g) for g in small_graphs]}")
    
    for graph_file in small_graphs:
        graph_name = os.path.basename(graph_file).replace('.txt', '')
        print(f"\n🔄 測試圖檔: {graph_name}")
        
        # 測試Config 4
        success_4 = run_single_test(graph_file, 4, cycle=200, trial=3)
        
        # 測試Config 5
        success_5 = run_single_test(graph_file, 5, cycle=200, trial=3)
        
        if success_4 and success_5:
            print(f"✅ {graph_name}: 兩個配置都測試成功")
        else:
            print(f"⚠️ {graph_name}: 部分測試失敗")
    
    print(f"\n🎉 快速測試完成!")
    print("💡 如果測試成功，可以執行完整版本:")
    print("   python batch_compare_config4_5.py")

def main():
    parser = argparse.ArgumentParser(description="Config 4 vs 5 快速測試")
    parser.add_argument('--full', action='store_true', help="執行完整測試（所有圖檔）")
    parser.add_argument('--cycle', type=int, default=200, help="測試週期數（預設200）")
    parser.add_argument('--trial', type=int, default=3, help="測試試驗數（預設3）")
    
    args = parser.parse_args()
    
    if args.full:
        print("執行完整測試...")
        # 調用完整版腳本
        from batch_compare_config4_5 import Config45Comparator
        comparator = Config45Comparator()
        comparator.run_comparison()
    else:
        quick_comparison_test()

if __name__ == "__main__":
    main() 