import subprocess
import os
import pandas as pd
from itertools import product
import csv
from tqdm import tqdm
import re

def extract_float(pattern, text):
    """從文本中提取浮點數"""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def run_apsa_timing_test(timing_strategy, eta_beta, target_drop, window_size):
    """
    測試不同的自適應觸發時機策略
    timing_strategy: 控制適應性調整的觸發時機
    """
    
    # 根據時機策略調整參數
    if timing_strategy == "conservative":
        # 保守策略：減少調整頻率
        eta_alpha = 0.0  # 不調整alpha
        eta_beta = eta_beta * 0.5  # 減少beta調整
        stall_threshold = 300  # 更大的停滯閾值
        monitoring_freq = 50  # 每50步監控一次而不是每10步
        
    elif timing_strategy == "moderate":
        # 中等策略：平衡調整
        eta_alpha = 0.0005
        eta_beta = eta_beta
        stall_threshold = 200
        monitoring_freq = 20
        
    elif timing_strategy == "aggressive":
        # 激進策略：頻繁調整（原始設定）
        eta_alpha = 0.001
        eta_beta = eta_beta
        stall_threshold = 100
        monitoring_freq = 10
        
    else:  # minimal策略
        # 最小干預：只做基本的停滯檢測
        eta_alpha = 0.0
        eta_beta = 0.0001  # 極小的調整
        stall_threshold = 500
        monitoring_freq = 100
    
    cmd = [
        "python3", "gpu_MAXCUT_var0708.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", "2",
        "--cycle", "1000",
        "--trial", "5",
        "--tau", "8",
        "--config", "5",  # Config 5 (ApSA)
        "--res", "2",
        "--l_scale", "0.1",
        "--d_scale", "0.1",
        "--n_scale", "0.2",
        "--eta_alpha", str(eta_alpha),
        "--eta_beta", str(eta_beta),
        "--target_drop", str(target_drop),
        "--window_size", str(window_size),
        "--stall_threshold", str(stall_threshold)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        # 解析結果
        cut_average = extract_float(r'Average cut: ([\d.\-eE]+)', output)
        cut_max = extract_float(r'Maximum cut: ([\d.\-eE]+)', output)
        cut_min = extract_float(r'Minimum cut: ([\d.\-eE]+)', output)
        time_average = extract_float(r'Average annealing time: ([\d.\-eE]+)', output)
        energy_average = extract_float(r'Average energy: ([\d.\-eE]+)', output)
        mean_ratio = extract_float(r'Average reachability \[%\]: ([\d.\-eE]+)', output)
        max_ratio = extract_float(r'Maximum reachability \[%\]: ([\d.\-eE]+)', output)
        
        # 穩定性評估
        is_stable = energy_average is not None and energy_average < 0
        energy_variability = "Low" if is_stable else "High"
        
        return {
            'timing_strategy': timing_strategy,
            'eta_alpha': eta_alpha,
            'eta_beta': eta_beta,
            'target_drop': target_drop,
            'window_size': window_size,
            'stall_threshold': stall_threshold,
            'monitoring_freq': monitoring_freq,
            'average_cut': cut_average,
            'maximum_cut': cut_max,
            'minimum_cut': cut_min,
            'average_annealing_time_ms': time_average,
            'average_energy': energy_average,
            'average_reachability_percent': mean_ratio,
            'maximum_reachability_percent': max_ratio,
            'is_stable_convergence': is_stable,
            'energy_variability': energy_variability,
            # 性能比較
            'energy_gap_vs_spsa': abs(energy_average + 3743) if energy_average is not None else float('inf'),
            'reachability_gap_vs_spsa': abs(max_ratio - 98.6) if max_ratio is not None else float('inf'),
            'improvement_vs_basic_psa': (max_ratio - 76.9) if max_ratio is not None else -float('inf')
        }
        
    except subprocess.TimeoutExpired:
        print(f"實驗超時: {timing_strategy}")
        return None
    except Exception as e:
        print(f"實驗失敗: {e}")
        return None

def main():
    """測試Config 5的不同優化策略"""
    # 創建結果目錄
    os.makedirs('./apsa_timing_optimization', exist_ok=True)
    
    print("Config 5 (ApSA) 時機優化測試")
    print("目標：找到穩定收斂的自適應觸發策略")
    print("=" * 60)
    
    # 測試不同的時機策略
    timing_strategies = ["minimal", "conservative", "moderate", "aggressive"]
    
    # 針對每種策略測試不同的參數組合
    eta_betas = [0.001, 0.002, 0.003]
    target_drops = [50, 100, 150]
    window_sizes = [70, 100, 150]
    
    all_results = []
    
    for strategy in timing_strategies:
        print(f"\n{'='*50}")
        print(f"測試策略: {strategy.upper()}")
        print(f"{'='*50}")
        
        strategy_results = []
        
        for eta_beta, target_drop, window_size in tqdm(
            list(product(eta_betas, target_drops, window_sizes)), 
            desc=f"{strategy} 策略"
        ):
            print(f"\n測試參數: eta_beta={eta_beta}, target_drop={target_drop}, window_size={window_size}")
            
            result = run_apsa_timing_test(strategy, eta_beta, target_drop, window_size)
            
            if result is not None:
                strategy_results.append(result)
                all_results.append(result)
                
                stable_status = "穩定" if result['is_stable_convergence'] else "不穩定"
                print(f"結果: 能量={result['average_energy']:.1f}, 可達性={result['maximum_reachability_percent']:.1f}% ({stable_status})")
                
                if result['is_stable_convergence']:
                    print(f"與SpSA差距: 能量差{result['energy_gap_vs_spsa']:.1f}, 可達性差{result['reachability_gap_vs_spsa']:.1f}%")
            else:
                print("實驗失敗")
        
        # 分析每種策略的結果
        if strategy_results:
            df_strategy = pd.DataFrame(strategy_results)
            stable_count = df_strategy['is_stable_convergence'].sum()
            total_count = len(strategy_results)
            
            print(f"\n{strategy} 策略總結:")
            print(f"穩定收斂率: {stable_count}/{total_count} ({stable_count/total_count*100:.1f}%)")
            
            if stable_count > 0:
                stable_results = df_strategy[df_strategy['is_stable_convergence'] == True]
                best_result = stable_results.loc[stable_results['maximum_reachability_percent'].idxmax()]
                
                print(f"最佳穩定結果:")
                print(f"  能量: {best_result['average_energy']:.1f}")
                print(f"  可達性: {best_result['maximum_reachability_percent']:.1f}%")
                print(f"  參數: eta_beta={best_result['eta_beta']}, target_drop={best_result['target_drop']}")
    
    # 整體分析
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # 按策略分組分析
        strategy_summary = df_all.groupby('timing_strategy').agg({
            'is_stable_convergence': 'sum',
            'average_energy': 'mean',
            'maximum_reachability_percent': 'mean',
            'energy_gap_vs_spsa': 'mean',
            'reachability_gap_vs_spsa': 'mean'
        }).round(2)
        
        strategy_summary['stability_rate'] = (
            strategy_summary['is_stable_convergence'] / 
            df_all.groupby('timing_strategy').size() * 100
        ).round(1)
        
        # 找出最佳策略
        stable_results = df_all[df_all['is_stable_convergence'] == True]
        
        # 保存結果
        excel_path = './apsa_timing_optimization/apsa_timing_optimization_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 所有結果
            df_all.to_excel(writer, sheet_name='所有結果', index=False)
            
            # 策略總結
            strategy_summary.to_excel(writer, sheet_name='策略總結')
            
            # 穩定結果排序
            if not stable_results.empty:
                stable_sorted = stable_results.sort_values('maximum_reachability_percent', ascending=False)
                stable_sorted.to_excel(writer, sheet_name='穩定結果排序', index=False)
                
                # 各策略最佳結果
                best_by_strategy = []
                for strategy in timing_strategies:
                    strategy_stable = stable_results[stable_results['timing_strategy'] == strategy]
                    if not strategy_stable.empty:
                        best = strategy_stable.loc[strategy_stable['maximum_reachability_percent'].idxmax()]
                        best_by_strategy.append(best)
                
                if best_by_strategy:
                    df_best = pd.DataFrame(best_by_strategy)
                    df_best.to_excel(writer, sheet_name='各策略最佳', index=False)
            
            # 按策略分組的詳細統計
            for strategy in timing_strategies:
                strategy_data = df_all[df_all['timing_strategy'] == strategy]
                strategy_data.to_excel(writer, sheet_name=f'{strategy}_詳細', index=False)
        
        print(f"\n結果已保存到: {excel_path}")
        print(f"總共測試 {len(all_results)} 個配置")
        
        # 顯示策略比較
        print("\n" + "="*60)
        print("策略比較總結:")
        print(strategy_summary)
        
        if not stable_results.empty:
            # 找出總體最佳結果
            best_overall = stable_results.loc[stable_results['maximum_reachability_percent'].idxmax()]
            
            print(f"\n總體最佳Config 5配置:")
            print(f"策略: {best_overall['timing_strategy']}")
            print(f"參數: eta_alpha={best_overall['eta_alpha']}, eta_beta={best_overall['eta_beta']}")
            print(f"      target_drop={best_overall['target_drop']}, window_size={best_overall['window_size']}")
            print(f"      stall_threshold={best_overall['stall_threshold']}")
            print(f"性能: 能量={best_overall['average_energy']:.1f}, 可達性={best_overall['maximum_reachability_percent']:.1f}%")
            print(f"與SpSA差距: 能量差{best_overall['energy_gap_vs_spsa']:.1f}, 可達性差{best_overall['reachability_gap_vs_spsa']:.1f}%")
            
            # 推薦最佳策略
            best_strategy = strategy_summary.loc[strategy_summary['stability_rate'].idxmax()]
            print(f"\n推薦策略: {best_strategy.name} (穩定率: {best_strategy['stability_rate']:.1f}%)")
            
        else:
            print("\n警告：沒有找到穩定收斂的配置，建議:")
            print("1. 進一步降低eta_beta值")
            print("2. 增加window_size和stall_threshold") 
            print("3. 考慮完全禁用自適應調整(eta_alpha=0, eta_beta=0)")
    
    else:
        print("沒有成功完成的實驗")

if __name__ == "__main__":
    main() 