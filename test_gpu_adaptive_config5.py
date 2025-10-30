#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU自適應Config 5測試腳本
展示增強版GPU自適應PSA演算法的效能改進

主要改進：
1. 能量計算移到GPU端
2. 自適應參數調整在GPU端執行
3. 停滯檢測和cluster-flip在GPU端處理
4. 大幅減少主機與GPU之間的資料傳輸
"""

import subprocess
import time
import sys
import os

def run_test(graph_file, config, tau=8, res=1, trial=10, cycle=1000):
    """執行單個測試"""
    cmd = [
        'python', 'gpu_MAXCUT_var0708.py',
        '--file_path', f'graph/{graph_file}',
        '--config', str(config),
        '--tau', str(tau),
        '--res', str(res), 
        '--trial', str(trial),
        '--cycle', str(cycle),
        '--l_scale', '0.1',
        '--d_scale', '0.1', 
        '--n_scale', '0.3',
        '--thread', '32'
    ]
    
    print(f"\n{'='*60}")
    print(f"測試 Config {config} - {graph_file}")
    print(f"參數: tau={tau}, res={res}, trial={trial}, cycle={cycle}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print("✅ 測試成功完成")
            print(f"執行時間: {end_time - start_time:.2f} 秒")
            
            # 提取關鍵結果
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['GPU Adaptive', 'Final alpha', 'Cut value', 'Ising Energy', 'Maximum cut']):
                    print(f"  {line}")
                    
        else:
            print("❌ 測試失敗")
            print("錯誤輸出:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏱️ 測試超時 (300秒)")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
    
    return end_time - start_time if 'end_time' in locals() else 0

def compare_configs():
    """比較不同config的效能"""
    graph_files = ['G1.txt', 'G55.txt']  # 選擇一些代表性的圖
    configs = [2, 5]  # 比較原始pSA和GPU自適應pSA
    results = {}
    
    print(f"\n{'='*80}")
    print("Config 2 (原始pSA) vs Config 5 (GPU自適應pSA) 效能比較")
    print(f"{'='*80}")
    
    for graph in graph_files:
        print(f"\n📊 圖檔: {graph}")
        results[graph] = {}
        
        for config in configs:
            config_name = "原始pSA" if config == 2 else "GPU自適應pSA"
            print(f"\n🔄 測試 {config_name} (Config {config})")
            
            execution_time = run_test(graph, config, tau=8, res=1, trial=5, cycle=500)
            results[graph][config] = execution_time
            
    # 總結比較結果
    print(f"\n{'='*80}")
    print("📈 效能比較總結")
    print(f"{'='*80}")
    
    for graph in graph_files:
        if 2 in results[graph] and 5 in results[graph]:
            original_time = results[graph][2]
            adaptive_time = results[graph][5]
            if original_time > 0 and adaptive_time > 0:
                speedup = original_time / adaptive_time
                print(f"\n📊 {graph}:")
                print(f"  原始pSA (Config 2):     {original_time:.2f} 秒")
                print(f"  GPU自適應pSA (Config 5): {adaptive_time:.2f} 秒")
                print(f"  加速比: {speedup:.2f}x")
                
                if speedup > 1.1:
                    print("  🚀 GPU自適應版本顯著更快!")
                elif speedup < 0.9:
                    print("  ⚠️ 需要進一步優化")
                else:
                    print("  📊 效能相當")

def test_adaptive_features():
    """測試自適應功能的有效性"""
    print(f"\n{'='*80}")
    print("🧪 GPU自適應功能測試")
    print(f"{'='*80}")
    
    # 測試不同的自適應參數
    test_cases = [
        {'eta_beta': 1e-3, 'target_drop': 50, 'window_size': 50},
        {'eta_beta': 1e-2, 'target_drop': 100, 'window_size': 100},
        {'eta_beta': 1e-4, 'target_drop': 20, 'window_size': 20}
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n🔬 測試案例 {i}: {params}")
        
        cmd = [
            'python', 'gpu_MAXCUT_var0708.py',
            '--file_path', 'graph/G1.txt',
            '--config', '5',
            '--tau', '8',
            '--res', '1',
            '--trial', '3',
            '--cycle', '300',
            '--eta_beta', str(params['eta_beta']),
            '--target_drop', str(params['target_drop']),
            '--window_size', str(params['window_size']),
            '--l_scale', '0.1',
            '--d_scale', '0.1',
            '--n_scale', '0.3'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("  ✅ 測試成功")
                # 提取GPU自適應資訊
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'GPU Adaptive' in line:
                        print(f"    {line}")
            else:
                print("  ❌ 測試失敗")
        except subprocess.TimeoutExpired:
            print("  ⏱️ 測試超時")

def main():
    """主測試函數"""
    print("🎯 GPU自適應Config 5測試程式")
    print("===============================")
    
    # 檢查必要檔案
    required_files = [
        'gpu_MAXCUT_var0708.py',
        'apsa_annealing_kernel_var.cu',
        'graph/G1.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少必要檔案: {missing_files}")
        return
    
    try:
        # 1. 基本功能測試
        print("\n1️⃣ 基本功能測試")
        run_test('G1.txt', 5, tau=8, res=1, trial=3, cycle=200)
        
        # 2. 效能比較測試
        print("\n2️⃣ 效能比較測試")
        compare_configs()
        
        # 3. 自適應功能測試
        print("\n3️⃣ 自適應功能測試")
        test_adaptive_features()
        
        print(f"\n{'='*80}")
        print("🎉 所有測試完成!")
        print("📝 詳細結果請查看上述輸出")
        print("📊 能量圖表已保存到 ./0708_energy_plots/ 目錄")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")

if __name__ == "__main__":
    main() 