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

def extract_int(pattern, text):
    """從文本中提取整數"""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None

def run_apsa_experiment(stall_threshold, eta_alpha, eta_beta, target_drop, window_size):
    """運行單個ApSA實驗"""
    cmd = [
        "python3", "gpu_MAXCUT_var0708.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", "2",  # 使用param 2
        "--cycle", "1000",
        "--trial", "5",  # 減少trial數以加快測試
        "--tau", "8",  # 使用tau 8
        "--config", "5",  # Config 5 (ApSA)
        "--res", "2",  # 使用res 2
        "--l_scale", "0.1",
        "--d_scale", "0.1", 
        "--n_scale", "0.1",
        "--stall_threshold", str(stall_threshold),
        "--eta_alpha", str(eta_alpha),
        "--eta_beta", str(eta_beta),
        "--target_drop", str(target_drop),
        "--window_size", str(window_size)
    ]
    
    try:
        # 執行並擷取輸出
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
        # 解析輸出指標
        cut_average = extract_float(r'Average cut: ([\d.\-eE]+)', output)
        cut_max = extract_float(r'Maximum cut: ([\d.\-eE]+)', output)
        cut_min = extract_float(r'Minimum cut: ([\d.\-eE]+)', output)
        time_average = extract_float(r'Average annealing time: ([\d.\-eE]+)', output)
        energy_average = extract_float(r'Average energy: ([\d.\-eE]+)', output)
        
        # TTS可能是None或數值
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
        
        return {
            'stall_threshold': stall_threshold,
            'eta_alpha': eta_alpha,
            'eta_beta': eta_beta,
            'target_drop': target_drop,
            'window_size': window_size,
            'average_cut': cut_average,
            'maximum_cut': cut_max,
            'minimum_cut': cut_min,
            'average_annealing_time_ms': time_average,
            'average_energy': energy_average,
            'tts_099': tts_099,
            'average_reachability_percent': mean_ratio,
            'maximum_reachability_percent': max_ratio
        }
        
    except subprocess.TimeoutExpired:
        print(f"實驗超時: stall_threshold={stall_threshold}, eta_alpha={eta_alpha}, eta_beta={eta_beta}, target_drop={target_drop}, window_size={window_size}")
        return None
    except Exception as e:
        print(f"實驗失敗: {e}")
        return None

def main():
    """主函數，執行所有參數組合的實驗"""
    # 創建結果目錄
    os.makedirs('./apsa_optimization_results', exist_ok=True)
    
    # 定義參數範圍
    stall_thresholds = [50, 100, 150]  # 停滯閾值
    eta_alphas = [0.0, 0.001, 0.002]  # alpha學習率
    eta_betas = [0.001, 0.002, 0.003, 0.004]  # beta學習率  
    target_drops = [60, 80, 100]  # 目標能量下降
    window_sizes = [50, 70, 90]  # 窗口大小
    
    # 生成所有參數組合
    param_combinations = list(product(stall_thresholds, eta_alphas, eta_betas, target_drops, window_sizes))
    
    print(f"總共將運行 {len(param_combinations)} 個實驗")
    
    # 儲存結果的列表
    results = []
    
    # 執行所有實驗
    for i, (stall_threshold, eta_alpha, eta_beta, target_drop, window_size) in enumerate(tqdm(param_combinations, desc="ApSA參數優化")):
        print(f"\n[{i+1}/{len(param_combinations)}] 測試參數組合:")
        print(f"  stall_threshold={stall_threshold}, eta_alpha={eta_alpha}, eta_beta={eta_beta}")
        print(f"  target_drop={target_drop}, window_size={window_size}")
        
        result = run_apsa_experiment(stall_threshold, eta_alpha, eta_beta, target_drop, window_size)
        
        if result is not None:
            results.append(result)
            # 安全格式化輸出，處理None值
            avg_cut = result['average_cut'] if result['average_cut'] is not None else "N/A"
            max_ratio = result['maximum_reachability_percent'] if result['maximum_reachability_percent'] is not None else "N/A"
            if avg_cut != "N/A" and max_ratio != "N/A":
                print(f"  結果: avg_cut={avg_cut:.1f}, max_ratio={max_ratio:.2f}%")
            else:
                print(f"  結果: avg_cut={avg_cut}, max_ratio={max_ratio}%")
        else:
            print("  實驗失敗或超時")
    
    # 將結果轉換為DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        # 按照最大可達性百分比排序
        df_sorted = df.sort_values('maximum_reachability_percent', ascending=False)
        
        # 保存到Excel文件
        excel_path = './apsa_optimization_results/apsa_parameter_optimization_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 原始數據
            df_sorted.to_excel(writer, sheet_name='所有結果', index=False)
            
            # 前10名結果
            df_top10 = df_sorted.head(10)
            df_top10.to_excel(writer, sheet_name='前10名結果', index=False)
            
            # 統計摘要
            summary_stats = df.describe()
            summary_stats.to_excel(writer, sheet_name='統計摘要')
            
        print(f"\n結果已保存到: {excel_path}")
        print(f"總共完成 {len(results)} 個實驗")
        print(f"最佳結果 (最大可達性): {df_sorted.iloc[0]['maximum_reachability_percent']:.2f}%")
        print("最佳參數組合:")
        best_result = df_sorted.iloc[0]
        print(f"  stall_threshold: {best_result['stall_threshold']}")
        print(f"  eta_alpha: {best_result['eta_alpha']}")
        print(f"  eta_beta: {best_result['eta_beta']}")
        print(f"  target_drop: {best_result['target_drop']}")
        print(f"  window_size: {best_result['window_size']}")
        
    else:
        print("沒有成功完成的實驗")

if __name__ == "__main__":
    main() 