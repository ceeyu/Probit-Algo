import pandas as pd
import glob

# 找出所有 summary 結果檔案
csv_files = glob.glob('./testvar_result/resultvar_summary.csv') + \
            glob.glob('./testvar_result0616/resultvar0616_summary_tau100.csv')

all_results = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        all_results.append(df)
    except Exception as e:
        print(f"讀取 {file} 時發生錯誤: {e}")

if not all_results:
    print("沒有找到任何 summary 結果檔案！")
    exit(0)

# 合併所有結果
summary = pd.concat(all_results, ignore_index=True)

# 排序找出 mean_ratio/max_ratio 靠近 100% 的組合
summary_sorted_mean = summary.sort_values(by='mean_ratio', ascending=False)
summary_sorted_max = summary.sort_values(by='max_ratio', ascending=False)

# 輸出前 20 名
print("\n--- mean_ratio 靠近 100% 的前 20 組合 ---")
print(summary_sorted_mean[['config','param','tau','res','l_scale','d_scale','n_scale','mean_range','stall_prop','mean_ratio','max_ratio']].head(20))

print("\n--- max_ratio 靠近 100% 的前 20 組合 ---")
print(summary_sorted_max[['config','param','tau','res','l_scale','d_scale','n_scale','mean_range','stall_prop','mean_ratio','max_ratio']].head(20))

# 也可以將合併排序後的結果存成新檔案
summary_sorted_mean.to_csv('./testvar_result/all_results_sorted_by_mean_ratio.csv', index=False)
summary_sorted_max.to_csv('./testvar_result/all_results_sorted_by_max_ratio.csv', index=False)
print("\n已將排序結果存成 all_results_sorted_by_mean_ratio.csv 及 all_results_sorted_by_max_ratio.csv") 