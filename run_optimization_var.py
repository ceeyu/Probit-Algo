import subprocess
import os
import pandas as pd
from itertools import product
import csv
from tqdm import tqdm

def write_summary_row(summary_path, row, header):
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_experiment(config, param, tau, res, l_scale, d_scale, n_scale, mean_range=None, stall_prop=None):
    cmd = [
        "python3", "gpu_MAXCUT_var.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", str(param),
        "--cycle", "1000",
        "--trial", "10",
        "--tau", str(tau),
        "--config", str(config),
        "--res", str(res),
        "--l_scale", str(l_scale),
        "--d_scale", str(d_scale),
        "--n_scale", str(n_scale)
    ]
    
    if mean_range is not None:
        cmd.extend(["--mean_range", str(mean_range)])
    if stall_prop is not None:
        cmd.extend(["--stall_prop", str(stall_prop)])
    
    # 執行並擷取輸出
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # 解析輸出
    def extract_float(pattern, text):
        import re
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    def extract_int(pattern, text):
        import re
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None
    
    cut_average = extract_float(r'Average cut: ([\d.\-eE]+)', output)
    cut_max = extract_float(r'Maximum cut: ([\d.\-eE]+)', output)
    time_average = extract_float(r'Average annealing time: ([\d.\-eE]+)', output)
    total_time = extract_float(r'Total time: ([\d.\-eE]+)', output)
    mean_ratio = extract_float(r'Average reachability \[%\]: ([\d.\-eE]+)', output)
    max_ratio = extract_float(r'Maximum reachability \[%\]: ([\d.\-eE]+)', output)
    
    # 寫入 summary
    summary_path = './testvar_result/resultvar_summary.csv'
    header = [
        'config', 'param', 'tau', 'res', 'l_scale', 'd_scale', 'n_scale',
        'mean_range', 'stall_prop', 'mean_ratio', 'max_ratio',
        'cut_average', 'cut_max', 'time_average', 'total_time'
    ]
    row = {
        'config': config,
        'param': param,
        'tau': tau,
        'res': res,
        'l_scale': l_scale,
        'd_scale': d_scale,
        'n_scale': n_scale,
        'mean_range': mean_range,
        'stall_prop': stall_prop,
        'mean_ratio': mean_ratio,
        'max_ratio': max_ratio,
        'cut_average': cut_average,
        'cut_max': cut_max,
        'time_average': time_average,
        'total_time': total_time
    }
    write_summary_row(summary_path, row, header)
    return mean_ratio, max_ratio

def main():
    # 創建結果目錄
    os.makedirs('./testvar_result', exist_ok=True)
    
    # 測試參數範圍
    params = [1, 2]  # param 值
    taus = [1, 2, 4, 8]  # tau 值
    mean_ranges = [2, 4, 8, 16]  # mean_range 值
    stall_props = [0.1, 0.3, 0.5, 0.7, 0.9]  # stall_prop 值
    
    # 設備變異性參數
    res_values = [1, 2, 4]  # 時間解析度
    l_scale_values = [0.1, 0.2, 0.3]  # lambda 標準差
    d_scale_values = [0.1, 0.2, 0.3]  # delta 標準差
    n_scale_values = [0.1, 0.2, 0.3]  # nu 標準差
    
    # 測試 config 2 (pSA)
    print("Testing config 2 (pSA)...")
    comb2 = list(product(params, taus, res_values, l_scale_values, d_scale_values, n_scale_values))
    for param, tau, res, l_scale, d_scale, n_scale in tqdm(comb2, desc="pSA"):
        mean_ratio, max_ratio = run_experiment(2, param, tau, res, l_scale, d_scale, n_scale)
        print(f"[pSA] param={param}, tau={tau}, res={res}, l_scale={l_scale}, d_scale={d_scale}, n_scale={n_scale} | mean_ratio={mean_ratio}, max_ratio={max_ratio}")
    
    # 測試 config 3 (TApSA)
    print("Testing config 3 (TApSA)...")
    comb3 = list(product(params, taus, res_values, l_scale_values, d_scale_values, n_scale_values, mean_ranges))
    for param, tau, res, l_scale, d_scale, n_scale, mean_range in tqdm(comb3, desc="TApSA"):
        mean_ratio, max_ratio = run_experiment(3, param, tau, res, l_scale, d_scale, n_scale, mean_range=mean_range)
        print(f"[TApSA] param={param}, tau={tau}, res={res}, l_scale={l_scale}, d_scale={d_scale}, n_scale={n_scale}, mean_range={mean_range} | mean_ratio={mean_ratio}, max_ratio={max_ratio}")
    
    # 測試 config 4 (SpSA)
    print("Testing config 4 (SpSA)...")
    comb4 = list(product(params, taus, res_values, l_scale_values, d_scale_values, n_scale_values, stall_props))
    for param, tau, res, l_scale, d_scale, n_scale, stall_prop in tqdm(comb4, desc="SpSA"):
        mean_ratio, max_ratio = run_experiment(4, param, tau, res, l_scale, d_scale, n_scale, stall_prop=stall_prop)
        print(f"[SpSA] param={param}, tau={tau}, res={res}, l_scale={l_scale}, d_scale={d_scale}, n_scale={n_scale}, stall_prop={stall_prop} | mean_ratio={mean_ratio}, max_ratio={max_ratio}")

if __name__ == "__main__":
    main() 