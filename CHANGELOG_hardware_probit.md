# CHANGELOG — `hardware_multiple_spin_probit_annealing.py`

> 作者：YU SHAO-HSIEN (shaoxianyou@gmail.com)  
> 紀錄範圍：所有與此檔案相關的 Git Commit（共 4 次更新）  
> 最後更新：2025-11-13

---

## 目錄

1. [v1 — 初版建立](#v1--初版建立-dbd3086--2025-11-12-1916)
2. [v2 — 硬體架構大重構](#v2--硬體架構大重構-4c82e90--2025-11-12-2325)
3. [v3 — 補充程式碼註解](#v3--補充程式碼註解-14aa976--2025-11-13-1654)
4. [v4 — 新增非同步算法與批次輸出支援](#v4--新增非同步算法與批次輸出支援-a86774e--2025-11-13-2233)
5. [函式演進總表](#函式演進總表)
6. [核心數學說明](#核心數學說明)

---

## v1 — 初版建立 (`dbd3086` | 2025-11-12 19:16)

**Commit 訊息：** `1112 加入G matrix改{-1,0,1}, spin改{0,1}檔案`  
**變更規模：** 新增檔案，共 757 行

### 概述

首次建立此 Python 腳本，實作 Max-Cut 問題的「Probit 類比退火法」與「傳統模擬退火法」比較實驗框架，模擬硬體 Crossbar 陣列上的退火行為。

---

### 新增函式一覽

#### `parse_arguments()`
- 解析命令列參數，支援以下選項：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--file_path` | 必填 | GSET 圖資料檔路徑 |
| `--trial` | 50 | 實驗重複次數 |
| `--timesteps` | 10000 | 退火迭代次數 |
| `--sigma_start` | 5.0 | Probit 起始噪聲值 |
| `--sigma_end` | 0.01 | Probit 結束噪聲值 |
| `--T_start` | 5.0 | SA 起始溫度 |
| `--T_end` | 0.01 | SA 結束溫度 |
| `--schedule` | `exponential` | 退火排程（`exponential` / `linear`）|
| `--probit_mode` | `synchronous` | Probit 更新模式（`synchronous` / `asynchronous`）|
| `--epsilon` | 0.1 | RPA 比例，每個 timestep 可更新的 spin 比例 |

---

#### `read_file_MAXCUT(file_path)`
- 讀取 GSET 格式圖檔案，回傳節點數、邊數、邊值類型、best-known 值及邊列表。

#### `get_graph_MAXCUT(vertex, lines)`
- 從邊列表建構對稱鄰接矩陣 `G_matrix`（`dtype=np.int32`）。
- GSET 節點編號從 1 開始，轉換成 0-indexed。
- 使用 `G_matrix + G_matrix.T` 完成對稱化。

#### `quantize_xbar_from_G(G_matrix)`
> **此函式於 v2 被移除**

- 將 GSET 原始權重 `G_matrix ∈ {-1, 0, +1}` 映射至硬體 Crossbar 可儲存權重：

  | G 值 | → | J_xbar 值 |
  |------|---|-----------|
  | -1   | → | 0.0       |
  |  0   | → | 0.5       |
  | +1   | → | 1.0       |

- 對角線強制設為 0（不使用自耦合）。

---

#### `calculate_energy(m_vector, J_matrix)`
- 計算 Ising 模型能量，公式與 `gpu_MAXCUT_var0708.py` 一致：

  ```
  h_vector = diag(J_matrix)
  J_energy = ((J @ m - h) · m) / 2
  h_energy = h · m
  energy   = -(J_energy + h_energy)
  ```

- **限制（v1）**：僅接受 Ising spin `{-1, +1}`，不支援 binary spin `{0, 1}`。

#### `calculate_cut(m_vector, G_matrix)`
- 計算 Max-Cut 值（向量化上三角法）：

  ```
  cut = Σ G[i,j] * (1 - s[i]*s[j]) / 2   (對上三角所有邊)
  ```

- **限制（v1）**：同樣僅支援 Ising spin `{-1, +1}`。

---

#### `probit_annealing_synchronous(J_matrix, ...)`
- 同步 RPA（Ratio-Controlled Parallel Annealing）Probit 退火。
- Spin 型態：Ising `{-1, +1}`。
- 每個 timestep 流程：
  1. **平行 MVM**：`I_vector = J @ m_vector`（模擬 Crossbar 一次平行輸出）。
  2. **全局雜訊**：產生一個共同 Gaussian noise，全 N 個 spin 共用。
  3. **平行決策**：`m_proposed = sign(I + noise)`。
  4. **RPA 遮罩**：
     - 找出所有「欲翻轉」的 spin index。
     - 若欲翻轉數 ≤ `N × epsilon`，全部允許翻轉。
     - 若欲翻轉數 > `N × epsilon`，隨機挑選 `N × epsilon` 個允許翻轉。
  5. **更新 spin**。

#### `probit_fitting_hardware_synchronous(J_matrix, ...)`
- 與 `probit_annealing_synchronous` 邏輯相似，RPA 遮罩改為「機率比較器」版本（LFSR 硬體對應）：
  - 為每個 spin 產生 uniform `[0,1)` 隨機數（對應 LFSR）。
  - 與 `epsilon` 比較，`< epsilon` 者才被允許更新（對應比較器）。

#### `probit_binary_spin_synchronous(J_mvm, J_eval, ...)`
> **此函式於 v2 大幅重構**

- Binary spin `{0, 1}` 版本的同步 Probit 退火。
- 使用量化後的硬體矩陣 `J_mvm ∈ {0, 0.5, 1}` 做 MVM。
- 需另外傳入 `J_eval`（理想 Ising 矩陣）用於能量計算。
- Binary 決策：`b = 1 if I + noise > 0 else 0`。
- 回傳前將 binary spin 轉回 Ising spin：`s = 2b - 1`。

#### `probit_annealing(J_matrix, ...)`
> **此函式於 v2 被移除，於 v4 重新加入**

- 非同步 Probit 退火（Monte Carlo Sweep，MCS）。
- 每個 timestep 執行 N 次：
  - 隨機選一個 spin `i`。
  - 計算局部場 `I_i = J[i,:] · m`。
  - 產生單一 Gaussian 雜訊。
  - 決策 `new_m = sign(I_i + noise)`。
  - 立即更新，並用 `ΔE = 2 × m_old × I_i` 增量更新能量。

#### `traditional_sa(J_matrix, ...)`
- 傳統 Metropolis-Hastings 模擬退火。
- Ising spin `{-1, +1}`，非同步單 spin 更新。
- 接受準則：`ΔE ≤ 0` 直接接受；`ΔE > 0` 以 `exp(-ΔE/T)` 機率接受。

---

#### `run_comparison_experiment(args, J_mvm, J_eval, G_matrix, best_known)`
> **v2 後介面簡化**

- 執行多次 trial，交替呼叫 Probit 與 SA。
- 收集並統計：能量、Cut 值、執行時間。
- 印出各統計量（平均、標準差、最小、最大、best-known 達成率）。

#### `save_results_and_plots(args, results, file_base, best_known)`
- 儲存以下結果：
  - **CSV**：統計摘要。
  - **PNG**：能量/Cut 值分佈直方圖、能量演化曲線（最後一次 trial）。
  - **Excel**：統計摘要 + 詳細逐次結果 + 能量演化序列（多 sheet）。
- 輸出目錄固定為 `./multiple_spin_probit_comparison_results`。

#### `main()`
- 讀取 GSET 檔案，建立：
  - `G_matrix`：原始圖。
  - `J_eval = -G_matrix`：用於能量/Cut 評估的 Ising 矩陣。
  - `J_xbar = quantize_xbar_from_G(G_matrix)`：硬體 Crossbar 矩陣。
- 呼叫 `run_comparison_experiment(args, J_xbar, J_eval, G_matrix, best_known)`。

---

## v2 — 硬體架構大重構 (`4c82e90` | 2025-11-12 23:25)

**Commit 訊息：** `1112 加入G matrix改{-1,0,1}, spin改{0,1}檔案，更新檔案在hardware裡面，1112,11:25`  
**變更規模：** +183 行 / -310 行（淨差約 -127 行）

### 概述

最大幅度的重構版本。核心目標是：

1. 在程式碼最前面加入完整的硬體行為文件說明。
2. 統一能量與 Cut 的計算接口，支援 binary / Ising spin 自動轉換。
3. 把硬體 Crossbar MVM 的量化轉換**內化到演算法函式內部**，移除外部的 `quantize_xbar_from_G` 與雙矩陣介面。
4. 簡化 `main()` 與 `run_comparison_experiment()` 的呼叫介面。

---

### A. 新增硬體架構說明文件區塊

在 `import` 區塊之後，加入大型 docstring，詳細說明：

```
硬體層級模擬 - Crossbar 權重轉換
```

| 主題 | 說明 |
|------|------|
| 硬體限制 | Crossbar 只能存 `{0, 0.5, 1}`，Spin 暫存器存 `{0, 1}` 或 `{-1, 1}` |
| 權重轉換 | `J_hw = (J + 1) / 2`，將 `{-1,0,1}` 映射至 `{0, 0.5, 1}` |
| Spin 轉換 | `b = (s + 1) / 2`，Ising → Binary |
| MVM 修正公式 | `I = 4·I_hw - 2·J_hw_row_sums - 2·b_sum + N` |
| 核心原則 | 能量與 Cut 計算永遠使用原始 `J_matrix` / `G_matrix`，硬體轉換對演算法透明 |

**完整推導過程**：

```
設 J_hw = (J+1)/2, b = (s+1)/2
I_hw = Σ_j J_hw_ij × b_j
     = Σ_j ((J_ij+1)/2) × ((s_j+1)/2)
     = (1/4) × [I + Σ_j J_ij + Σ_j s_j + N]

因此：
I = 4·I_hw - Σ_j J_ij - Σ_j s_j - N

對 binary spin（s = 2b-1，s_sum = 2·b_sum - N）：
I = 4·I_hw - 2·J_hw_row_sums - 2·b_sum + N
```

---

### B. `get_graph_MAXCUT` 微調

- 擴充 docstring，明確說明回傳的是**原始** GSET 權重 `{-1, 0, 1}`，硬體轉換由演算法內部處理。
- 新增診斷輸出：

  ```python
  print(f'[圖形權重] 範圍 [{np.min(G_matrix)}, {np.max(G_matrix)}]')
  ```

---

### C. 移除 `quantize_xbar_from_G`

- 此函式被完全刪除。
- 硬體量化邏輯改由 `probit_fitting_hardware_synchronous` 內部的預計算取代（見下方 D）。

---

### D. `calculate_energy` 泛化（自動偵測 spin 型態）

**舊版**（v1）：只接受 Ising spin `{-1, +1}`。

**新版**：自動偵測 spin 型態後轉換：

```python
def calculate_energy(spin_vector_input, J_matrix):
    if np.all((spin_vector_input == 0) | (spin_vector_input == 1)):
        s_vector = 2 * spin_vector_input - 1   # Binary {0,1} → Ising {-1,+1}
    elif np.all((spin_vector_input == -1) | (spin_vector_input == 1)):
        s_vector = spin_vector_input            # 已是 Ising {-1,+1}
    else:
        s_vector = 2 * spin_vector_input - 1   # 容許浮點誤差
    # 之後統一用 s_vector 計算能量...
```

此改動使 binary spin 函式的回傳值可直接傳入能量計算，**不再需要呼叫方手動轉換**。

---

### E. `calculate_cut` 泛化（自動偵測 spin 型態）

與 `calculate_energy` 相同邏輯，自動把 `{0,1}` 轉成 `{-1,+1}` 再計算 Cut 值。

---

### F. `probit_fitting_hardware_synchronous` 大改版

**介面變更**：  
舊：`probit_binary_spin_synchronous(J_mvm, J_eval, ...)`  
新：`probit_fitting_hardware_synchronous(J_matrix, ...)`

| 變更點 | 舊版 (v1) | 新版 (v2) |
|--------|-----------|-----------|
| MVM 矩陣來源 | 外部傳入 `J_mvm ∈ {0,0.5,1}` | 函式**內部**從 `J_matrix` 計算 `J_hw = (J+1)/2` |
| row sum 預計算 | 無 | 進入迴圈前預先計算 `J_hw_row_sums` |
| 硬體 MVM | `I = J_mvm @ b` | `I_hw = J_hw @ b`，再套修正公式還原 `I` |
| 修正公式 | 無 | `I = 4·I_hw - 2·J_hw_row_sums - 2·b_sum + N` |
| 能量評估矩陣 | 需額外傳 `J_eval` | 直接用 `J_matrix` |
| 回傳型態 | 轉回 Ising spin `{-1,+1}` | 直接回傳 binary spin `{0,1}` |
| spin 初始化 | `{-1,+1}` | `{0, 1}` |

**RPA 遮罩對應說明**（強化硬體描述）：

```
硬體對應：
  LFSR_1...LFSR_n → np.random.rand(N)       （產生 uniform [0,1) 隨機數）
  比較器           → update_mask = rand < epsilon  （決定是否允許更新）
  邏輯閘           → np.where(mask, proposed, current)  （最終決策）
```

---

### G. 移除 `probit_annealing`（非同步 MCS 版）

> 此版本暫時移除非同步更新功能，於 v4 重新加入。

---

### H. `run_comparison_experiment` 介面簡化

**舊**：`run_comparison_experiment(args, J_mvm, J_eval, G_matrix, best_known)`  
**新**：`run_comparison_experiment(args, J_matrix, G_matrix, best_known)`

- 同步模式改呼叫 `probit_fitting_hardware_synchronous`（取代 `probit_binary_spin_synchronous`）。
- 所有能量計算統一使用 `J_matrix`。
- 移除「硬體 Xbar 權重集合 Jij ∈ {0, 0.5, 1}」的 print 說明（因已在內部處理）。

---

### I. `main()` 介面簡化

**舊**：
```python
J_eval = -G_matrix
J_xbar = quantize_xbar_from_G(G_matrix)
results = run_comparison_experiment(args, J_xbar, J_eval, G_matrix, best_known)
```

**新**：
```python
J_matrix = -G_matrix
results = run_comparison_experiment(args, J_matrix, G_matrix, best_known)
```

只維護一個矩陣 `J_matrix = -G_matrix`，硬體轉換完全由演算法函式負責。

---

## v3 — 補充程式碼註解 (`14aa976` | 2025-11-13 16:54)

**Commit 訊息：** `1113 加入G matrix改{-1,0,1}, spin改{0,1}檔案，更新檔案在hardware`  
**變更規模：** +2 行 / -2 行（純文字變更，無邏輯修改）

### 概述

本次 commit **只增加行內中文註解**，幫助閱讀者理解變數意義，不涉及任何演算法或邏輯變動。

---

### 修改明細

**位置 1**：`probit_fitting_hardware_synchronous` 第一行

```python
# 修改前
N = J_matrix.shape[0]

# 修改後
N = J_matrix.shape[0]  # 如果 J 是一個100*100的矩陣，則J_matrix.shape會回傳(100,100)
```

**位置 2**：`probit_fitting_hardware_synchronous` 數位修正電路區塊

```python
# 修改前
b_sum = np.sum(b_vector)

# 修改後
b_sum = np.sum(b_vector)  # 用硬體參數結果還原比較值
```

---

## v4 — 新增非同步算法與批次輸出支援 (`a86774e` | 2025-11-13 22:33)

**Commit 訊息：** `1113 加入script檔案，自動比較Gset`  
**變更規模：** +70 行 / -1 行

### 概述

主要完成兩件事：

1. **重新加入非同步 MCS Probit 退火**（`probit_annealing`），整合進 v2 統一後的能量計算框架。
2. **支援批次自動化實驗**：輸出目錄可透過環境變數動態指定。

---

### A. 重新加入 `probit_annealing`（非同步 MCS）

> 此函式在 v2 被移除，v4 以新版架構重新加入。

```python
def probit_annealing(J_matrix, timesteps, sigma_start, sigma_end,
                     schedule='linear', record_energy=False):
```

**演算法流程**（每個 timestep 執行 N 次 MCS）：

```
for t in range(timesteps):
    sigma = annealing_schedule[t]
    for _ in range(N):
        i = 隨機選一個 spin index
        I_i = dot(J_matrix[i,:], m_vector)     # 局部場
        noise_i = normal(0, sigma)              # 單一雜訊
        new_m_i = sign(I_i + noise_i)           # 決策
        if new_m_i == 0: new_m_i = 1           # 處理邊界
        if m_vector[i] != new_m_i:
            ΔE = 2 × m_old × I_i               # 增量能量更新
            current_energy += ΔE
            m_vector[i] = new_m_i
    if record_energy:
        energy_history.append(current_energy)
```

與 v1 版本的差異：

| 比較項目 | v1 版本 | v4 版本 |
|----------|---------|---------|
| 能量計算函式 | 不支援 binary spin | 支援（繼承 v2 泛化版） |
| 架構定位 | 獨立函式 | 整合進統一 `J_matrix` 介面 |
| 能量增量更新 | 完全相同 | 完全相同 |
| 退火排程 | 完全相同 | 完全相同 |

---

### B. `save_results_and_plots` 支援環境變數輸出目錄

```python
# 舊版（v1/v2/v3）
output_dir = './multiple_spin_probit_comparison_results'

# 新版（v4）
output_dir = os.environ.get('HARDWARE_OUTPUT_DIR',
                            './multiple_spin_probit_comparison_results')
```

**用途**：配合同一 commit 加入的自動化 script 使用。script 在執行多組 GSET 比較時，可透過設定環境變數 `HARDWARE_OUTPUT_DIR` 將每批次的結果分別存放到不同資料夾，而不需要修改 Python 程式碼本身。

---

## 函式演進總表

| 函式名稱 | v1 (dbd3086) | v2 (4c82e90) | v3 (14aa976) | v4 (a86774e) |
|----------|:---:|:---:|:---:|:---:|
| `parse_arguments` | ✅ 新增 | — | — | — |
| `read_file_MAXCUT` | ✅ 新增 | — | — | — |
| `get_graph_MAXCUT` | ✅ 新增 | 🔄 加 print | — | — |
| `quantize_xbar_from_G` | ✅ 新增 | ❌ 移除 | — | — |
| `calculate_energy` | ✅ 新增 | 🔄 泛化（自動偵測 spin 型態） | — | — |
| `calculate_cut` | ✅ 新增 | 🔄 泛化（自動偵測 spin 型態） | — | — |
| `probit_annealing_synchronous` | ✅ 新增 | — | — | — |
| `probit_fitting_hardware_synchronous` | ✅ 新增 | 🔄 大改版（內化硬體 MVM） | 🔄 加註解 | — |
| `probit_binary_spin_synchronous` | ✅ 新增 | ❌ 整合至上方函式 | — | — |
| `probit_annealing`（非同步 MCS） | ✅ 新增 | ❌ 移除 | — | ✅ 重新加入 |
| `traditional_sa` | ✅ 新增 | — | — | — |
| `run_comparison_experiment` | ✅ 新增 | 🔄 介面簡化 | — | — |
| `save_results_and_plots` | ✅ 新增 | — | — | 🔄 支援環境變數 |
| `main` | ✅ 新增 | 🔄 介面簡化 | — | — |

圖示說明：✅ 新增 ｜ 🔄 修改 ｜ ❌ 移除 ｜ — 無變更

---

## 核心數學說明

### Ising 能量公式

$$E = -\left(\frac{1}{2}\sum_{i \ne j} J_{ij} s_i s_j + \sum_i h_i s_i\right)$$

其中 $h_i = J_{ii}$（對角線元素），$s_i \in \{-1, +1\}$。

### Max-Cut 值計算

$$\text{Cut} = \sum_{(i,j) \in E} w_{ij} \cdot \frac{1 - s_i s_j}{2}$$

### Crossbar MVM 修正公式

$$I = 4 I_\text{hw} - 2 \sum_j J^\text{hw}_{ij} - 2 b_\text{sum} + N$$

其中：
- $I_\text{hw} = J_\text{hw} \cdot b$（硬體實際輸出）
- $J^\text{hw}_{ij} = \frac{J_{ij} + 1}{2} \in \{0, 0.5, 1\}$
- $b_j = \frac{s_j + 1}{2} \in \{0, 1\}$
- $b_\text{sum} = \sum_j b_j$
- $N$ = spin 總數

### Probit 決策規則

**同步（Ising spin）**：

$$m_i^\text{new} = \text{sgn}\!\left(\sum_j J_{ij} m_j + \epsilon_\text{noise}\right), \quad \epsilon_\text{noise} \sim \mathcal{N}(0, \sigma^2)$$

**同步（Binary spin，硬體版）**：

$$b_i^\text{new} = \mathbf{1}\!\left[I_i + \epsilon_\text{noise} > 0\right]$$

### 退火排程

**指數（Exponential）**：

$$\sigma_t = \sigma_\text{start} \cdot \left(\frac{\sigma_\text{end}}{\sigma_\text{start}}\right)^{t/T}$$

**線性（Linear）**：

$$\sigma_t = \sigma_\text{start} + \frac{t}{T-1}(\sigma_\text{end} - \sigma_\text{start})$$

---

*此 CHANGELOG 由 Cursor AI 自動依 `git diff` 結果生成，最後更新時間：2026-02-26。*
