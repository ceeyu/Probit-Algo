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

def create_simplified_apsa_config(eta_alpha, eta_beta, target_drop, window_size, stall_threshold):
    """
    創建簡化版ApSA配置
    策略：移除tau循環內的頻繁同步，只在tau循環外做簡單的參數調整
    """
    return {
        'eta_alpha': eta_alpha,
        'eta_beta': eta_beta, 
        'target_drop': target_drop,
        'window_size': window_size,
        'stall_threshold': stall_threshold
    }

def run_simplified_apsa_experiment(config_dict, base_param, tau, res, l_scale, d_scale, n_scale):
    """運行簡化版ApSA實驗"""
    cmd = [
        "python3", "gpu_MAXCUT_var0708.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt", 
        "--param", str(base_param),
        "--cycle", "1000",
        "--trial", "5",
        "--tau", str(tau),
        "--config", "5",  # Config 5 (ApSA) - 但使用簡化參數
        "--res", str(res),
        "--l_scale", str(l_scale),
        "--d_scale", str(d_scale),
        "--n_scale", str(n_scale),
        # 簡化的自適應參數 - 設定較保守的值避免過度干擾
        "--eta_alpha", str(config_dict['eta_alpha']),
        "--eta_beta", str(config_dict['eta_beta']),
        "--target_drop", str(config_dict['target_drop']),
        "--window_size", str(config_dict['window_size']),
        "--stall_threshold", str(config_dict['stall_threshold'])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        output = result.stdout
        
        # 解析輸出指標
        cut_average = extract_float(r'Average cut: ([\d.\-eE]+)', output)
        cut_max = extract_float(r'Maximum cut: ([\d.\-eE]+)', output)
        cut_min = extract_float(r'Minimum cut: ([\d.\-eE]+)', output)
        time_average = extract_float(r'Average annealing time: ([\d.\-eE]+)', output)
        energy_average = extract_float(r'Average energy: ([\d.\-eE]+)', output)
        
        tts_match = re.search(r'TTS\(0\.99\): ([^\n]+)', output)
        tts_099 = tts_match.group(1).strip() if tts_match else None
        if tts_099 == "None":
            tts_099 = None
        else:
            try:
                tts_099 = float(tts_099)
            except:
                tts_099 = None
        
        mean_ratio = extract_float(r'Average reachability \[%\]: ([\d.\-eE]+)', output)
        max_ratio = extract_float(r'Maximum reachability \[%\]: ([\d.\-eE]+)', output)
        
        # 檢查是否穩定收斂（能量為負值）
        is_stable = energy_average is not None and energy_average < 0
        
        return {
            **config_dict,  # 包含所有ApSA配置參數
            'base_param': base_param,
            'tau': tau,
            'res': res,
            'l_scale': l_scale,
            'd_scale': d_scale,
            'n_scale': n_scale,
            'average_cut': cut_average,
            'maximum_cut': cut_max,
            'minimum_cut': cut_min,
            'average_annealing_time_ms': time_average,
            'average_energy': energy_average,
            'tts_099': tts_099,
            'average_reachability_percent': mean_ratio,
            'maximum_reachability_percent': max_ratio,
            'is_stable_convergence': is_stable,
            # 與SpSA和基礎pSA的比較
            'energy_gap_vs_spsa': abs(energy_average + 3743) if energy_average is not None else float('inf'),
            'reachability_gap_vs_spsa': abs(max_ratio - 98.6) if max_ratio is not None else float('inf'),
            'energy_gap_vs_psa': abs(energy_average - 1305) if energy_average is not None else float('inf'),
            'reachability_improvement_vs_psa': (max_ratio - 76.9) if max_ratio is not None else -float('inf')
        }
        
    except subprocess.TimeoutExpired:
        print(f"實驗超時")
        return None
    except Exception as e:
        print(f"實驗失敗: {e}")
        return None

def main():
    """主函數，執行簡化ApSA優化實驗"""
    # 創建結果目錄
    os.makedirs('./simplified_apsa_results', exist_ok=True)
    
    print("目標：修復並簡化ApSA，使其穩定收斂並逼近SpSA性能")
    print("SpSA目標性能：能量-3743，可達性98.6%")
    print("基礎pSA性能：能量1305，可達性76.9%")
    print("=" * 60)
    
    # 定義簡化的自適應參數範圍 - 避免過度調整
    eta_alphas = [0.0, 0.0005, 0.001]  # 較小的學習率，0.0表示不使用alpha調整
    eta_betas = [0.0005, 0.001, 0.002]  # 較保守的beta調整
    target_drops = [20, 50, 100]  # 較溫和的目標下降
    window_sizes = [50, 100, 150]  # 較大的窗口以減少頻繁調整
    stall_thresholds = [100, 200, 300]  # 較大的停滯閾值
    
    # 基礎參數
    base_params = [1, 2]
    taus = [8, 12, 16]  # 中等範圍的tau值
    res_values = [2, 4]  # 中等解析度
    
    # 設備變異性參數 - 使用較溫和的範圍
    l_scales = [0.0, 0.1, 0.2]
    d_scales = [0.0, 0.1, 0.2]
    n_scales = [0.2, 0.4]
    
    print("第一階段：測試核心自適應參數組合")
    
    # 創建ApSA配置組合
    apsa_configs = []
    for eta_alpha, eta_beta, target_drop, window_size, stall_threshold in product(
        eta_alphas, eta_betas, target_drops, window_sizes, stall_thresholds
    ):
        # 跳過可能過度激進的組合
        if eta_alpha > 0.001 and eta_beta > 0.002:
            continue
        if target_drop > 100 and window_size < 100:
            continue
            
        config = create_simplified_apsa_config(
            eta_alpha, eta_beta, target_drop, window_size, stall_threshold
        )
        apsa_configs.append(config)
    
    print(f"將測試 {len(apsa_configs)} 種ApSA配置")
    
    # 第一階段：測試核心自適應參數
    phase1_results = []
    
    for i, config in enumerate(tqdm(apsa_configs[:20], desc="Phase 1: 核心ApSA配置")):  # 先測試前20個
        print(f"\n[Phase1 {i+1}/20] 測試ApSA配置:")
        print(f"  eta_alpha={config['eta_alpha']}, eta_beta={config['eta_beta']}")
        print(f"  target_drop={config['target_drop']}, window_size={config['window_size']}")
        print(f"  stall_threshold={config['stall_threshold']}")
        
        # 使用中等參數進行測試
        result = run_simplified_apsa_experiment(
            config, base_param=2, tau=8, res=2, 
            l_scale=0.1, d_scale=0.1, n_scale=0.2
        )
        
        if result is not None:
            phase1_results.append(result)
            stable_status = "穩定" if result['is_stable_convergence'] else "不穩定"
            print(f"  結果: 能量={result['average_energy']:.1f}, 可達性={result['maximum_reachability_percent']:.1f}% ({stable_status})")
            
            if result['is_stable_convergence']:
                energy_gap = result['energy_gap_vs_spsa']
                reachability_gap = result['reachability_gap_vs_spsa']
                print(f"  與SpSA差距: 能量差{energy_gap:.1f}, 可達性差{reachability_gap:.1f}%")
        else:
            print("  實驗失敗")
    
    # 分析第一階段結果
    if phase1_results:
        df_phase1 = pd.DataFrame(phase1_results)
        
        # 先篩選出穩定收斂的配置
        stable_configs = df_phase1[df_phase1['is_stable_convergence'] == True]
        
        if not stable_configs.empty:
            print(f"\n發現 {len(stable_configs)} 個穩定收斂的ApSA配置！")
            
            # 對穩定配置進行評分
            stable_configs = stable_configs.copy()
            stable_configs['combined_score'] = (
                (stable_configs['energy_gap_vs_spsa'] / 4000) * 0.5 +
                (stable_configs['reachability_gap_vs_spsa'] / 20) * 0.5
            )
            
            best_stable = stable_configs.nsmallest(3, 'combined_score')
            
            print("\n最佳穩定ApSA配置（前3名）：")
            for idx, row in best_stable.iterrows():
                print(f"配置: eta_alpha={row['eta_alpha']}, eta_beta={row['eta_beta']}")
                print(f"參數: target_drop={row['target_drop']}, window_size={row['window_size']}")
                print(f"性能: 能量={row['average_energy']:.1f}, 可達性={row['maximum_reachability_percent']:.1f}%")
                print(f"評分: {row['combined_score']:.3f}")
                print("-" * 50)
            
            # 第二階段：使用最佳配置測試不同的基礎參數
            print("\n第二階段：使用最佳ApSA配置測試基礎參數組合")
            
            best_config = best_stable.iloc[0]
            best_apsa_config = {
                'eta_alpha': best_config['eta_alpha'],
                'eta_beta': best_config['eta_beta'],
                'target_drop': best_config['target_drop'],
                'window_size': int(best_config['window_size']),
                'stall_threshold': int(best_config['stall_threshold'])
            }
            
            print(f"使用最佳ApSA配置: {best_apsa_config}")
            
            # 測試不同基礎參數組合
            phase2_combinations = list(product(base_params, taus, res_values, l_scales, d_scales, n_scales))
            
            phase2_results = []
            
            for i, (param, tau, res, l_scale, d_scale, n_scale) in enumerate(tqdm(phase2_combinations[:30], desc="Phase 2: 基礎參數")):
                print(f"\n[Phase2 {i+1}/30] 測試基礎參數:")
                print(f"  param={param}, tau={tau}, res={res}")
                print(f"  l_scale={l_scale}, d_scale={d_scale}, n_scale={n_scale}")
                
                result = run_simplified_apsa_experiment(
                    best_apsa_config, param, tau, res, l_scale, d_scale, n_scale
                )
                
                if result is not None:
                    phase2_results.append(result)
                    stable_status = "穩定" if result['is_stable_convergence'] else "不穩定"
                    print(f"  結果: 能量={result['average_energy']:.1f}, 可達性={result['maximum_reachability_percent']:.1f}% ({stable_status})")
            
            # 合併結果並分析
            all_results = phase1_results + phase2_results
            df_all = pd.DataFrame(all_results)
            
            # 篩選穩定結果
            stable_all = df_all[df_all['is_stable_convergence'] == True]
            
            if not stable_all.empty:
                stable_all = stable_all.copy()
                stable_all['combined_score'] = (
                    (stable_all['energy_gap_vs_spsa'] / 4000) * 0.5 +
                    (stable_all['reachability_gap_vs_spsa'] / 20) * 0.5
                )
                
                # 保存結果
                excel_path = './simplified_apsa_results/simplified_apsa_optimization_results.xlsx'
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # 穩定配置按評分排序
                    stable_sorted = stable_all.sort_values('combined_score')
                    stable_sorted.to_excel(writer, sheet_name='穩定收斂配置', index=False)
                    
                    # 最佳可達性
                    best_reachability = stable_all.nlargest(10, 'maximum_reachability_percent')
                    best_reachability.to_excel(writer, sheet_name='最佳可達性', index=False)
                    
                    # 最佳能量
                    best_energy = stable_all.nsmallest(10, 'average_energy')
                    best_energy.to_excel(writer, sheet_name='最佳能量', index=False)
                    
                    # 與基礎pSA比較
                    improved_vs_psa = stable_all[stable_all['reachability_improvement_vs_psa'] > 0]
                    if not improved_vs_psa.empty:
                        improved_vs_psa.to_excel(writer, sheet_name='相比pSA改善', index=False)
                    
                    # 接近SpSA性能
                    close_to_spsa = stable_all[
                        (stable_all['energy_gap_vs_spsa'] < 1500) |
                        (stable_all['reachability_gap_vs_spsa'] < 10)
                    ]
                    if not close_to_spsa.empty:
                        close_to_spsa.to_excel(writer, sheet_name='接近SpSA性能', index=False)
                    
                    # 所有結果（包括不穩定的）
                    df_all.to_excel(writer, sheet_name='所有結果', index=False)
                
                print(f"\n結果已保存到: {excel_path}")
                print(f"發現 {len(stable_all)} 個穩定的ApSA配置")
                
                # 顯示最終最佳結果
                best_final = stable_sorted.iloc[0]
                print("\n" + "="*60)
                print("最終最佳簡化ApSA結果：")
                print(f"ApSA配置:")
                print(f"  eta_alpha: {best_final['eta_alpha']}")
                print(f"  eta_beta: {best_final['eta_beta']}")  
                print(f"  target_drop: {best_final['target_drop']}")
                print(f"  window_size: {best_final['window_size']}")
                print(f"  stall_threshold: {best_final['stall_threshold']}")
                print(f"基礎參數:")
                print(f"  param: {best_final['base_param']}")
                print(f"  tau: {best_final['tau']}")
                print(f"  res: {best_final['res']}")
                print(f"  l_scale: {best_final['l_scale']}")
                print(f"  d_scale: {best_final['d_scale']}")
                print(f"  n_scale: {best_final['n_scale']}")
                print(f"性能表現:")
                print(f"  能量: {best_final['average_energy']:.1f} (SpSA: -3743, 原pSA: 1305)")
                print(f"  可達性: {best_final['maximum_reachability_percent']:.1f}% (SpSA: 98.6%, 原pSA: 76.9%)")
                
                # 計算改善程度
                if best_final['reachability_improvement_vs_psa'] > 0:
                    improvement = (best_final['reachability_improvement_vs_psa'] / (98.6 - 76.9)) * 100
                    print(f"  相對於原pSA改善程度: {improvement:.1f}%")
                
            else:
                print("沒有找到穩定收斂的ApSA配置")
        else:
            print("第一階段沒有找到穩定收斂的配置，建議調整參數範圍")
    else:
        print("沒有成功完成的實驗")

if __name__ == "__main__":
    main() 