import subprocess
import os
import pandas as pd
from itertools import product

def run_experiment(config, param, tau, mean_range=None, stall_prop=None):
    cmd = [
        "python3", "gpu_MAXCUT.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", str(param),
        "--cycle", "1000",
        "--trial", "10",
        "--tau", str(tau),
        "--config", str(config)
    ]
    
    if mean_range is not None:
        cmd.extend(["--mean_range", str(mean_range)])
    if stall_prop is not None:
        cmd.extend(["--stall_prop", str(stall_prop)])
    
    subprocess.run(cmd)
    
    # 讀取結果
    if config == 2:
        alg = 'pSA'
    elif config == 3:
        alg = 'TApSA'
    elif config == 4:
        alg = 'SpSA'
        
    csv_file = f'./result/{alg}_result_unique1_config{config}_cycle1000_trial10_tau{tau}_thread32_param{param}.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df['ratio of mean/best'].iloc[-1], df['ratio of max/best'].iloc[-1]
    return None, None

def main():
    # 創建結果目錄
    os.makedirs('./test_result', exist_ok=True)
    
    # 測試參數範圍
    params = [1, 2]  # param 值
    taus = [1, 2, 4, 8]  # tau 值
    mean_ranges = [2, 4, 8, 16]  # mean_range 值
    stall_props = [0.1, 0.3, 0.5, 0.7, 0.9]  # stall_prop 值
    
    results = []
    
    # 測試 config 2 (pSA)
    print("Testing config 2 (pSA)...")
    for param, tau in product(params, taus):
        mean_ratio, max_ratio = run_experiment(2, param, tau)
        results.append({
            'config': 2,
            'param': param,
            'tau': tau,
            'mean_ratio': mean_ratio,
            'max_ratio': max_ratio
        })
    
    # 測試 config 3 (TApSA)
    print("Testing config 3 (TApSA)...")
    for param, tau, mean_range in product(params, taus, mean_ranges):
        mean_ratio, max_ratio = run_experiment(3, param, tau, mean_range=mean_range)
        results.append({
            'config': 3,
            'param': param,
            'tau': tau,
            'mean_range': mean_range,
            'mean_ratio': mean_ratio,
            'max_ratio': max_ratio
        })
    
    # 測試 config 4 (SpSA)
    print("Testing config 4 (SpSA)...")
    for param, tau, stall_prop in product(params, taus, stall_props):
        mean_ratio, max_ratio = run_experiment(4, param, tau, stall_prop=stall_prop)
        results.append({
            'config': 4,
            'param': param,
            'tau': tau,
            'stall_prop': stall_prop,
            'mean_ratio': mean_ratio,
            'max_ratio': max_ratio
        })
    
    # 將結果保存到CSV文件
    df = pd.DataFrame(results)
    df.to_csv('./test_result/optimization_results.csv', index=False)
    
    # 找出最佳結果
    best_mean = df.loc[df['mean_ratio'].idxmax()]
    best_max = df.loc[df['max_ratio'].idxmax()]
    
    print("\n最佳平均比率結果:")
    print(best_mean)
    print("\n最佳最大比率結果:")
    print(best_max)

if __name__ == "__main__":
    main() 