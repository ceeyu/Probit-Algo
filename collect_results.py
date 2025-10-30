import pandas as pd
import glob
import re
import os

csv_files = glob.glob('./result/*_result_*.csv')

all_results = []

for file in csv_files:
    filename = os.path.basename(file)
    config = re.search(r'config(\\d+)', filename)
    param = re.search(r'param(\\d+)', filename)
    tau = re.search(r'tau(\\d+)', filename)
    res = re.search(r'res(\\d+)', filename)
    config = int(config.group(1)) if config else None
    param = int(param.group(1)) if param else None
    tau = int(tau.group(1)) if tau else None
    res = int(res.group(1)) if res else None

    try:
        df = pd.read_csv(file)
        last_row = df.tail(1)
        mean_ratio = last_row['ratio of mean/best'].iloc[0] if 'ratio of mean/best' in last_row else None
        max_ratio = last_row['ratio of max/best'].iloc[0] if 'ratio of max/best' in last_row else None
        mean_range = last_row['mean_range'].iloc[0] if 'mean_range' in last_row else None
        stall_prop = last_row['stall_prop'].iloc[0] if 'stall_prop' in last_row else None
        l_scale = last_row['l_scale'].iloc[0] if 'l_scale' in last_row else None
        d_scale = last_row['d_scale'].iloc[0] if 'd_scale' in last_row else None
        n_scale = last_row['n_scale'].iloc[0] if 'n_scale' in last_row else None
        all_results.append({
            'config': config,
            'param': param,
            'tau': tau,
            'res': res,
            'l_scale': l_scale,
            'd_scale': d_scale,
            'n_scale': n_scale,
            'mean_ratio': mean_ratio,
            'max_ratio': max_ratio,
            'mean_range': mean_range,
            'stall_prop': stall_prop
        })
    except Exception as e:
        print(f"讀取 {file} 時發生錯誤: {e}")

# 指定欄位順序
col_order = ['config', 'param', 'tau', 'res', 'l_scale', 'd_scale', 'n_scale', 'mean_ratio', 'max_ratio', 'mean_range', 'stall_prop']
summary = pd.DataFrame(all_results)
summary = summary[col_order]
summary.to_csv('./result/summary_all_results.csv', index=False)
print('已彙整所有結果到 ./result/summary_all_results.csv')