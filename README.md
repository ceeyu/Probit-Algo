# GPU-pSAv: GPU加速並行模擬退火變異演算法

一個用於求解最大切割問題(Max-Cut Problem)的GPU並行模擬退火演算法集合，支持多種變異和自適應策略。

## 概述

GPU-pSAv實現了多種並行模擬退火演算法的GPU加速版本，特別針對最大切割問題進行優化。項目包含從基礎SSA到先進自適應演算法的完整演算法族。

### 主要特點

✅ **GPU並行加速**：利用CUDA實現高效並行計算  
✅ **多種演算法配置**：支持8種不同的退火策略  
✅ **設備變異性支持**：模擬真實硬體的參數變異  
✅ **自適應機制**：動態調整參數以改善收斂性能  
✅ **完整的實驗框架**：包含參數優化和結果分析工具

---

## 演算法配置總覽

| Config | 演算法名稱 | 特點 | 穩定性 | 推薦用途 |
|--------|------------|------|---------|-----------|
| **1** | SSA | 基礎同步模擬退火 | 高 | 基準測試 |
| **2** | pSA | 並行模擬退火 | 中等 | 一般應用 |
| **3** | TApSA | 時間平均並行退火 | 高 | 高品質解 |
| **4** | SpSA | 停滯感知並行退火 | **優秀** | **生產環境推薦** |
| **5** | ApSA | 自適應並行退火 | 不穩定 | 研究實驗 |
| **6** | HpSA | 混合策略退火 | 中等 | 多樣化搜索 |
| **7** | MpSA | 多族群並行退火 | 中等 | 複雜問題 |
| **8** | SmartpSA | 智能自適應退火 | 中等 | 高級應用 |

### 性能基準 (G1圖, 800頂點)

| 演算法 | 平均能量 | 可達性 | 穩定性 | 計算時間 |
|--------|----------|---------|--------|----------|
| **SpSA (Config 4)** | **-3743** | **98.6%** | 優秀 | 中等 |
| TApSA (Config 3) | -7500~-8000 | 85-90% | 高 | 稍長 |
| pSA (Config 2) | 1305 | 76.9% | 中等 | 快 |
| ApSA (Config 5) | 639 | 79.7% | 差 | 變化大 |

---

## 快速開始

### 環境要求

- **CUDA環境**：CUDA 11.0+
- **Python**：3.8+
- **依賴套件**：
  ```bash
  pip install numpy pandas matplotlib tqdm openpyxl pycuda
  ```

### 基本使用

#### 1. 運行SpSA（推薦配置）
```bash
python3 gpu_MAXCUT_var0708.py \
    --gpu 0 \
    --file_path "./graph/G1.txt" \
    --param 2 \
    --cycle 1000 \
    --trial 10 \
    --tau 8 \
    --config 4 \
    --res 2 \
    --l_scale 0.1 \
    --d_scale 0.1 \
    --n_scale 0.2 \
    --stall_prop 0.5
```

#### 2. 運行標準pSA
```bash
python3 gpu_MAXCUT_var0708.py \
    --gpu 0 \
    --file_path "./graph/G1.txt" \
    --param 2 \
    --cycle 1000 \
    --trial 10 \
    --tau 8 \
    --config 2 \
    --res 2 \
    --l_scale 0.1 \
    --d_scale 0.1 \
    --n_scale 0.2
```

#### 3. 運行自適應ApSA
```bash
python3 gpu_MAXCUT_var0708.py \
    --gpu 0 \
    --file_path "./graph/G1.txt" \
    --param 2 \
    --cycle 1000 \
    --trial 10 \
    --tau 8 \
    --config 5 \
    --res 2 \
    --l_scale 0.1 \
    --d_scale 0.1 \
    --n_scale 0.2 \
    --eta_alpha 0.001 \
    --eta_beta 0.002 \
    --target_drop 80 \
    --window_size 70 \
    --stall_threshold 200
```

---

## 參數說明

### 基礎參數

| 參數 | 類型 | 說明 | 預設值 |
|------|------|------|--------|
| `--gpu` | int | GPU設備編號 | 0 |
| `--file_path` | str | 圖文件路徑 | 必填 |
| `--param` | int | 參數類型 (1或2) | 2 |
| `--cycle` | int | 退火週期數 | 1000 |
| `--trial` | int | 試驗次數 | 10 |
| `--tau` | int | 每溫度迭代次數 | 8 |
| `--config` | int | 演算法配置 (1-8) | 2 |
| `--res` | int | 時間解析度 | 2 |

### 設備變異性參數

| 參數 | 類型 | 說明 | 建議值 |
|------|------|------|--------|
| `--l_scale` | float | Lambda標準差 | 0.0-0.3 |
| `--d_scale` | float | Delta標準差 | 0.0-0.3 |
| `--n_scale` | float | Nu標準差 | 0.1-0.5 |

### 演算法特定參數

#### Config 3 (TApSA)
- `--mean_range`: 時間平均範圍 (預設: 4)

#### Config 4 (SpSA)
- `--stall_prop`: 停滯比例 (預設: 0.5)

#### Config 5 (ApSA)
- `--eta_alpha`: Alpha學習率 (預設: 1e-3)
- `--eta_beta`: Beta學習率 (預設: 1e-3)
- `--target_drop`: 目標能量下降 (預設: 100)
- `--window_size`: 監控窗口大小 (預設: 100)
- `--stall_threshold`: 停滯閾值 (預設: 200)

---

## 圖文件格式

### 支持的圖格式
```
頂點數
標籤類型
圖類型  
邊數
起點 終點 權重
起點 終點 權重
...
```

### 範例 (G1.txt)
```
800
unipolar
random
11624
1 560 1
1 503 1
1 264 1
...
```

---

## 結果輸出

### 控制台輸出
```
######################## Final result #######################
Average cut: 11459.666666666666
Maximum cut: 11505
Minimum cut: 11414
Average annealing time: 281.69740600585936 [ms]
Average energy: -3743.3333333333335
TTS(0.99): 281.69740600585936
Average reachability [%]: 98.58625831612754
Maximum reachability [%]: 99.21259842519685
```

### CSV文件輸出
- `./result/{演算法}_result_{參數}.csv`: 詳細結果數據
- `./result/{演算法}_cut_{參數}.csv`: 切割值記錄

---

## 參數優化工具

### 1. 簡化ApSA優化
```bash
python3 run_simplified_apsa_optimization.py
```
自動測試ApSA的穩定參數配置。

### 2. 時機優化測試
```bash
python3 run_apsa_timing_optimization.py
```
測試不同自適應觸發策略的效果。

---

## 詳細文檔

### Config 5 (ApSA) 詳細說明
請參閱 [`CONFIG5_ApSA_演算法詳解.md`](./CONFIG5_ApSA_演算法詳解.md) 了解：
- 演算法原理和動機
- 詳細實現流程
- 參數調優指南
- 已知問題和解決方案
- 最佳實踐建議

### 能量圖表比較工具
請參閱原README內容了解如何使用能量圖表比較工具。

---

## 實驗建議

### 新手用戶
1. 從**Config 4 (SpSA)**開始，它提供最穩定的性能
2. 使用預設參數進行初始測試
3. 根據結果逐步調整`stall_prop`參數

### 進階用戶
1. 嘗試**Config 3 (TApSA)**以獲得更高品質解
2. 實驗不同的設備變異性參數組合
3. 根據圖的特性調整`tau`和`cycle`參數

### 研究用戶
1. 探索**Config 5 (ApSA)**的自適應機制
2. 使用參數優化工具進行系統性實驗
3. 開發新的自適應策略

---

## 常見問題

### Q: 為什麼Config 5性能不穩定？
A: Config 5 (ApSA)在tau循環內進行頻繁的GPU同步，導致性能問題。建議使用保守參數或選擇Config 4。

### Q: 如何選擇最佳配置？
A: 對於大多數應用，推薦Config 4 (SpSA)。如需要最高品質解，可嘗試Config 3 (TApSA)。

### Q: 設備變異性參數如何調整？
A: 從小值開始 (l_scale=0.1, d_scale=0.1, n_scale=0.2)，根據結果逐步調整。

### Q: 如何判斷收斂品質？
A: 主要看能量值和可達性。穩定的負能量值和高可達性表示良好收斂。

---

## 項目結構

```
GPU-pSAv/
├── gpu_MAXCUT_var0708.py          # 主程序
├── CONFIG5_ApSA_演算法詳解.md      # Config 5詳細文檔
├── README.md                      # 本文檔
├── graph/                         # 測試圖文件
│   ├── G1.txt
│   ├── G55.txt
│   └── ...
├── result/                        # 結果輸出目錄
├── *.cu                          # CUDA kernel文件
├── run_*.py                      # 參數優化腳本
└── compare_energy_plots*.py      # 圖表比較工具
```

---

## 引用

如果您在研究中使用了這個項目，請引用相關論文。

---

## 授權

本項目採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 文件。
