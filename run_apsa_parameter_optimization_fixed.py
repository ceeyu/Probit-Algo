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

def run_psa_experiment(param, tau, res, l_scale, d_scale, n_scale):
    """運行標準pSA實驗（使用Config 2代替有問題的Config 5）"""
    cmd = [
        "python3", "gpu_MAXCUT_var0708.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", str(param),
        "--cycle", "1000",
                 "--trial", "5",  # 減少trial數以加快測試
        "--tau", str(tau),
        "--config", "2",  # 使用Config 2 (標準pSA)
        "--res", str(res),
        "--l_scale", str(l_scale),
        "--d_scale", str(d_scale),
        "--n_scale", str(n_scale)
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
            'param': param,
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
            'maximum_reachability_percent': max_ratio
        }
        
    except subprocess.TimeoutExpired:
        print(f"實驗超時: param={param}, tau={tau}, res={res}")
        return None
    except Exception as e:
        print(f"實驗失敗: {e}")
        return None

def main():
    """主函數，執行pSA參數優化實驗"""
    # 創建結果目錄
    os.makedirs('./psa_optimization_results', exist_ok=True)
    
    # 定義參數範圍 - 重點測試設備變異性參數
    params = [1, 2]  # 參數類型
    taus = [4, 8, 16]  # tau值
    res_values = [1, 2, 4]  # 時間解析度
    l_scale_values = [0.0, 0.1, 0.2, 0.3]  # lambda標準差
    d_scale_values = [0.0, 0.1, 0.2, 0.3]  # delta標準差  
    n_scale_values = [0.1, 0.3, 0.5]  # nu標準差
    
    # 生成所有參數組合
    param_combinations = list(product(params, taus, res_values, l_scale_values, d_scale_values, n_scale_values))
    
    print(f"總共將運行 {len(param_combinations)} 個pSA實驗")
    
    # 儲存結果的列表
    results = []
    
    # 執行所有實驗
    for i, (param, tau, res, l_scale, d_scale, n_scale) in enumerate(tqdm(param_combinations, desc="pSA參數優化")):
        print(f"\n[{i+1}/{len(param_combinations)}] 測試參數組合:")
        print(f"  param={param}, tau={tau}, res={res}")
        print(f"  l_scale={l_scale}, d_scale={d_scale}, n_scale={n_scale}")
        
        result = run_psa_experiment(param, tau, res, l_scale, d_scale, n_scale)
        
        if result is not None:
            results.append(result)
            # 安全格式化輸出，處理None值
            avg_cut = result['average_cut'] if result['average_cut'] is not None else "N/A"
            max_ratio = result['maximum_reachability_percent'] if result['maximum_reachability_percent'] is not None else "N/A"
            avg_energy = result['average_energy'] if result['average_energy'] is not None else "N/A"
            
            if avg_cut != "N/A" and max_ratio != "N/A":
                print(f"  結果: avg_cut={avg_cut:.1f}, max_ratio={max_ratio:.2f}%, energy={avg_energy}")
            else:
                print(f"  結果: avg_cut={avg_cut}, max_ratio={max_ratio}%, energy={avg_energy}")
        else:
            print("  實驗失敗或超時")
    
    # 將結果轉換為DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        # 按照最大可達性百分比排序
        df_sorted = df.sort_values('maximum_reachability_percent', ascending=False)
        
        # 保存到Excel文件
        excel_path = './psa_optimization_results/psa_parameter_optimization_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 原始數據
            df_sorted.to_excel(writer, sheet_name='所有結果', index=False)
            
            # 前10名結果
            df_top10 = df_sorted.head(10)
            df_top10.to_excel(writer, sheet_name='前10名結果', index=False)
            
            # 按能量排序的結果（最低能量最好）
            df_energy_sorted = df.sort_values('average_energy', ascending=True)
            df_energy_sorted.head(10).to_excel(writer, sheet_name='最佳能量結果', index=False)
            
            # 統計摘要
            summary_stats = df.describe()
            summary_stats.to_excel(writer, sheet_name='統計摘要')
            
        print(f"\n結果已保存到: {excel_path}")
        print(f"總共完成 {len(results)} 個實驗")
        print(f"最佳結果 (最大可達性): {df_sorted.iloc[0]['maximum_reachability_percent']:.2f}%")
        print("最佳參數組合 (可達性):")
        best_result = df_sorted.iloc[0]
        print(f"  param: {best_result['param']}")
        print(f"  tau: {best_result['tau']}")
        print(f"  res: {best_result['res']}")
        print(f"  l_scale: {best_result['l_scale']}")
        print(f"  d_scale: {best_result['d_scale']}")
        print(f"  n_scale: {best_result['n_scale']}")
        
        # 顯示最佳能量結果
        best_energy_result = df_energy_sorted.iloc[0]
        print(f"\n最佳能量結果: {best_energy_result['average_energy']:.2f}")
        print("最佳參數組合 (能量):")
        print(f"  param: {best_energy_result['param']}")
        print(f"  tau: {best_energy_result['tau']}")
        print(f"  res: {best_energy_result['res']}")
        print(f"  l_scale: {best_energy_result['l_scale']}")
        print(f"  d_scale: {best_energy_result['d_scale']}")
        print(f"  n_scale: {best_energy_result['n_scale']}")
        
    else:
        print("沒有成功完成的實驗")

if __name__ == "__main__":
    main() 