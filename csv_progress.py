import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取兩個 summary
df2 = pd.read_csv('testvar_result0616/resultvar0618_summary_config-2.csv')
df34 = pd.read_csv('testvar_result0616/resultvar0618_config-34summary.csv')
df = pd.concat([df2, df34], ignore_index=True)

# 只取每個 graph/config 的最佳 mean_ratio
best = df.loc[df.groupby(['graph_file', 'config'])['mean_ratio'].idxmax()]

# 畫圖
plt.figure(figsize=(12,6))
sns.lineplot(data=best, x='graph_file', y='mean_ratio', hue='config', marker='o')
plt.xticks(rotation=45)
plt.title('不同 Graph 下 config2/3/4 的最佳 mean_ratio')
plt.tight_layout()
plt.show()