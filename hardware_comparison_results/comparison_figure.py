import pandas as pd
import matplotlib.pyplot as plt
import re

# === 設定 Excel 檔案名稱 ===
file_path = "trial 1000 steps 10000 總比較表.xlsx"
# 若要畫 1000 版本 → 換成：
# file_path = "trial 100 steps 1000 總比較表.xlsx"

# === 讀取 Excel (無表頭模式) ===
df = pd.read_excel(file_path, header=None)

groups = {}
current = None

# === 找出每個 G-set 的 Probit & Traditional SA data ===
for i, row in df.iterrows():
    first = str(row[0])

    # 偵測 G-set 標題
    m = re.match(r"trial 1000 steps \d+\((G\d+)\)", first)
    if m:
        current = m.group(1)
        groups[current] = {}

    # 在 "Algorithm" 後的兩行是 Probit & Traditional SA
    elif first.strip() == "Algorithm" and current is not None:
        probit_row = df.iloc[i + 1]
        trad_row = df.iloc[i + 2]
        groups[current]["Probit"] = probit_row
        groups[current]["Traditional"] = trad_row

# === 找 Mean Accuracy (%) 欄位的 index ===
acc_idx = None
for i, row in df.iterrows():
    if str(row[0]).strip() == "Algorithm":
        header = df.iloc[i]
        for idx, val in header.items():
            if str(val).strip() == "Mean Accuracy (%)":
                acc_idx = idx
                break
        break

if acc_idx is None:
    raise ValueError("找不到 'Mean Accuracy (%)' 欄位！")

# === 收集 G1～G81 精準度資料 ===
labels = []
acc_probit = []
acc_trad = []

for g, vals in groups.items():
    labels.append(g)
    acc_probit.append(vals["Probit"][acc_idx])
    acc_trad.append(vals["Traditional"][acc_idx])

# === 畫圖 ===
plt.figure(figsize=(12, 6))
plt.plot(labels, acc_probit, label="Probit", linewidth=2)
plt.plot(labels, acc_trad, label="Traditional SA", linewidth=2)

plt.xlabel("G-set")
plt.ylabel("Mean Accuracy (%)")
plt.title("Mean Accuracy Comparison (Trial 1000, Steps10000)")
plt.xticks(rotation=90)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
