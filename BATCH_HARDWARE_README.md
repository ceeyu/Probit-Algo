# 硬體 Probit 類比退火法批次測試系統

## 系統概述

這個批次測試系統用於自動化執行大量的硬體 Probit 類比退火法比較實驗，並將結果按照參數組合分類存儲。

## 檔案說明

### 核心檔案

| 檔案 | 說明 |
|------|------|
| `hardware_multiple_spin_probit_annealing.py` | 主要演算法實現（已修改支援環境變數） |
| `batch_hardware_comparison.py` | 批次執行腳本（核心） |
| `批量硬體比較使用說明.md` | 詳細使用說明 |

### 測試與預覽工具

| 檔案 | 說明 |
|------|------|
| `test_batch_hardware.py` | 快速測試腳本（G1~G3） |
| `preview_batch_hardware.py` | Dry-run 預覽腳本 |

## 快速開始

### 步驟 1：預覽執行計劃（建議）

```bash
# 預覽 G1~G3 的執行計劃
python preview_batch_hardware.py --start 1 --end 3
```

這會顯示所有要執行的命令，但不實際執行。

### 步驟 2：快速測試（建議）

```bash
# 測試 G1~G3（約 5-10 分鐘）
python test_batch_hardware.py
```

這會執行小範圍測試，確認系統運作正常。

### 步驟 3：正式執行

```bash
# 執行完整測試（G1~G81，約 5.5 小時）
python batch_hardware_comparison.py

# 或執行部分測試（例如 G1~G20）
python batch_hardware_comparison.py --start_graph 1 --end_graph 20
```

## 測試參數

系統會自動測試以下 4 種參數組合：

| 組合 | Trial | Timestep | 說明 |
|------|-------|----------|------|
| 1 | 100 | 1,000 | 低試驗次數 × 低步數 |
| 2 | 100 | 10,000 | 低試驗次數 × 高步數 |
| 3 | 1,000 | 100 | 高試驗次數 × 低步數 |
| 4 | 1,000 | 10,000 | 高試驗次數 × 高步數 |

## 輸出結構

```
hardware_comparison_results/
├── trial100_steps1000/      # 組合 1 的所有結果
│   ├── comparison_G1_*.csv
│   ├── histogram_G1_*.png
│   ├── energy_evolution_G1_*.png
│   ├── detailed_results_G1_*.xlsx
│   ├── ... (G2~G81)
│   └── ...
├── trial100_steps10000/     # 組合 2 的所有結果
├── trial1000_steps100/      # 組合 3 的所有結果
└── trial1000_steps10000/    # 組合 4 的所有結果
```

## 常用命令範例

### 1. 預覽完整執行計劃

```bash
python preview_batch_hardware.py --start 1 --end 81
```

### 2. 測試小範圍（G1~G5）

```bash
python batch_hardware_comparison.py --start_graph 1 --end_graph 5
```

### 3. 使用線性退火排程

```bash
python batch_hardware_comparison.py --schedule linear
```

### 4. 調整 RPA 更新比例

```bash
python batch_hardware_comparison.py --epsilon 0.2
```

### 5. 使用非同步更新模式

```bash
python batch_hardware_comparison.py --probit_mode asynchronous
```

## 執行時間估算

| 範圍 | 實驗數 | 預計時間 |
|------|--------|----------|
| G1~G3 | 12 | 約 5-10 分鐘 |
| G1~G5 | 20 | 約 15-20 分鐘 |
| G1~G20 | 80 | 約 1-1.5 小時 |
| G1~G81 | 324 | 約 5-6 小時 |

*註：實際時間取決於圖檔大小和硬體效能*

## 結果分析

### 每個實驗產生的檔案

1. **CSV 摘要** (`comparison_*.csv`)
   - 統計比較：平均值、標準差、最小值、最大值
   - 達成率：相對於 best-known cut 的百分比

2. **直方圖** (`histogram_*.png`)
   - 能量分佈直方圖
   - Cut 值分佈直方圖

3. **能量演化曲線** (`energy_evolution_*.png`)
   - Probit vs Traditional SA 的能量變化

4. **Excel 詳細結果** (`detailed_results_*.xlsx`)
   - Sheet 1: 統計摘要
   - Sheet 2: 每次試驗的詳細結果
   - Sheet 3: 能量演化數據

### 如何比較不同參數組合

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取不同參數組合的結果
df1 = pd.read_csv('hardware_comparison_results/trial100_steps1000/comparison_G1_*.csv')
df2 = pd.read_csv('hardware_comparison_results/trial100_steps10000/comparison_G1_*.csv')
df3 = pd.read_csv('hardware_comparison_results/trial1000_steps100/comparison_G1_*.csv')
df4 = pd.read_csv('hardware_comparison_results/trial1000_steps10000/comparison_G1_*.csv')

# 比較平均 Cut 值
configs = ['100×1K', '100×10K', '1000×100', '1000×10K']
probit_cuts = [df1['Mean Cut'][0], df2['Mean Cut'][0], df3['Mean Cut'][0], df4['Mean Cut'][0]]
sa_cuts = [df1['Mean Cut'][1], df2['Mean Cut'][1], df3['Mean Cut'][1], df4['Mean Cut'][1]]

# 繪圖
plt.figure(figsize=(10, 6))
x = range(len(configs))
plt.bar([i-0.2 for i in x], probit_cuts, width=0.4, label='Probit', alpha=0.8)
plt.bar([i+0.2 for i in x], sa_cuts, width=0.4, label='Traditional SA', alpha=0.8)
plt.xticks(x, configs)
plt.xlabel('Configuration (Trial × Timestep)')
plt.ylabel('Mean Cut Value')
plt.title('Performance Comparison Across Different Configurations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 修改測試參數

如果需要修改測試參數組合，編輯 `batch_hardware_comparison.py`：

```python
def create_output_directories(base_dir='./hardware_comparison_results'):
    configs = [
        ('trial100_steps1000', 100, 1000),
        ('trial100_steps10000', 100, 10000),
        ('trial1000_steps100', 1000, 100),
        ('trial1000_steps10000', 1000, 10000),
        # 可以新增或修改參數組合
        # ('trial500_steps5000', 500, 5000),
    ]
    # ... 其餘程式碼 ...
```

## 注意事項

### 1. 磁碟空間

- 每個實驗約 1-5 MB
- G1~G81 完整測試約需 1-2 GB

### 2. 記憶體

- 大型圖檔（如 G55~G81）可能需要較多記憶體
- 建議至少 8 GB RAM

### 3. 執行中斷

- 目前不支援中斷後繼續
- 建議使用 `nohup` 或 `screen` 在背景執行長時間任務

```bash
# 使用 nohup（輸出會寫入 nohup.out）
nohup python batch_hardware_comparison.py > batch_log.txt 2>&1 &

# 或使用 screen（可以重新連接）
screen -S batch_hardware
python batch_hardware_comparison.py
# 按 Ctrl+A, D 離開（程序繼續執行）
# 重新連接：screen -r batch_hardware
```

### 4. 平行執行

目前是序列執行（一次一個實驗）。如果想要平行化：

```python
# 可以使用 multiprocessing 或分多台機器執行
# 例如：機器 1 執行 G1~G27，機器 2 執行 G28~G54，機器 3 執行 G55~G81
```

## 常見問題

### Q1: 找不到圖檔？
檢查 `./graph` 目錄是否存在，圖檔命名是否正確（`G1.txt`, `G2.txt`, ...）

### Q2: 執行失敗？
1. 先執行 `test_batch_hardware.py` 測試
2. 檢查 `hardware_multiple_spin_probit_annealing.py` 是否可單獨執行
3. 查看終端輸出的錯誤訊息

### Q3: 結果在哪裡？
所有結果在 `hardware_comparison_results/` 目錄下，按參數組合分類

### Q4: 如何停止執行？
按 `Ctrl+C`（會顯示失敗的實驗列表）

## 技術細節

### 硬體模擬

- **Crossbar 權重轉換**: J_hw = (J + 1) / 2 ∈ {0, 0.5, 1}
- **Spin 暫存器**: binary spin {0, 1}
- **MVM 修正電路**: I = 4*I_hw - 2*J_hw_row_sums - 2*b_sum + N
- **RPA 策略**: 使用 epsilon 控制每個 timestep 的更新比例

### 環境變數機制

批次腳本通過環境變數 `HARDWARE_OUTPUT_DIR` 告訴主程式輸出目錄：

```python
# 批次腳本設置
env['HARDWARE_OUTPUT_DIR'] = output_dir

# 主程式讀取
output_dir = os.environ.get('HARDWARE_OUTPUT_DIR', './multiple_spin_probit_comparison_results')
```

## 進階用法

### 1. 自定義參數範圍

```bash
# 測試特定範圍的圖檔
python batch_hardware_comparison.py --start_graph 10 --end_graph 30

# 使用不同的退火參數
python batch_hardware_comparison.py \
    --sigma_start 10.0 \
    --sigma_end 0.001 \
    --T_start 10.0 \
    --T_end 0.001
```

### 2. 只測試部分參數組合

修改 `batch_hardware_comparison.py` 的 `configs` 列表，註解掉不需要的組合。

### 3. 批次結果統計分析

可以創建額外的分析腳本來彙總所有結果：

```python
# analyze_all_results.py
import pandas as pd
from pathlib import Path

base_dir = Path('./hardware_comparison_results')

for config_dir in base_dir.iterdir():
    if config_dir.is_dir():
        print(f"\n{config_dir.name}:")
        csv_files = list(config_dir.glob('comparison_*.csv'))
        
        probit_accuracies = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            probit_accuracies.append(df['Max Accuracy (%)'][0])
        
        print(f"  平均最大達成率: {sum(probit_accuracies)/len(probit_accuracies):.2f}%")
```

## 支援與回饋

如有問題或建議，請參考：
- 詳細說明：`批量硬體比較使用說明.md`
- 原始程式：`hardware_multiple_spin_probit_annealing.py`

## 版本資訊

- **v1.0** (2025-01-13)
  - 初始版本
  - 支援 4 種參數組合自動化測試
  - 支援 G1~G81 批量處理
  - 結果分類存儲

