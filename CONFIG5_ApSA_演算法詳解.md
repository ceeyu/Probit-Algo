# Config 5: ApSA (Adaptive Parallel Simulated Annealing) 演算法詳解

## 概述

ApSA (Adaptive Parallel Simulated Annealing) 是基於標準pSA演算法的自適應改進版本，旨在通過動態調整退火參數來提升收斂性能和結果品質。該演算法在GPU並行環境下運行，專門用於求解最大切割問題(Max-Cut Problem)。

---

## 演算法動機

### 問題背景
標準的pSA演算法在處理不同規模和結構的圖時，往往需要手動調整參數來獲得最佳性能。特別是：

1. **固定參數的局限性**：不同圖結構需要不同的退火策略
2. **收斂速度問題**：在復雜圖上可能收斂緩慢或陷入局部最優
3. **穩定性挑戰**：參數設置不當會導致能量振盪

### ApSA的解決方案
ApSA通過以下自適應機制來解決這些問題：

- **動態溫度調整**：根據收斂狀況調整退火速度
- **自適應擾動控制**：根據能量變化調整搜索強度  
- **智能停滯檢測**：檢測並處理收斂停滯狀況
- **基於圖結構的參數縮放**：根據圖的大小和密度調整參數

---

## 演算法架構

### 核心組件

```
ApSA 演算法 = 標準pSA + 自適應監控機制 + 參數調整策略
```

#### 1. **基礎pSA核心**
- 使用標準的GPU並行退火kernel (`apsa_annealing_kernel_var.cu`)
- 保持與原始pSA相同的spin更新機制
- 支持設備變異性參數 (lambda, delta, nu)

#### 2. **自適應監控系統**
- **能量窗口追蹤**：維護近期能量變化的滑動窗口
- **收斂狀態檢測**：監控能量改善趨勢
- **停滯時間計算**：追蹤自上次改善以來的週期數

#### 3. **參數調整引擎**
- **Alpha調整**：控制搜索強度 (`current_alpha`)
- **Beta調整**：控制退火速度 (`current_beta`) 
- **Lambda適應**：動態調整擾動參數

---

## 詳細演算法流程

### 初始化階段

```python
# 1. 基礎參數設置
vertex_count = 圖的頂點數
tau = 每次溫度下的迭代次數
cycle = 總退火週期數

# 2. 自適應參數初始化
best_energy = float('inf')
last_improve_cycle = 0
energy_window = deque(maxlen=window_size)
current_alpha = 0.8  # 初始搜索強度
current_beta = beta  # 初始退火速度

# 3. 圖大小自適應縮放
if vertex_count <= 50:    # 小圖
    adaptive_target_drop = target_drop * 0.05
    adaptive_eta_beta = eta_beta * 2.0
    adaptive_window_size = window_size // 4
elif vertex_count <= 200: # 中圖  
    adaptive_target_drop = target_drop * 0.3
    adaptive_eta_beta = eta_beta * 1.2
    adaptive_window_size = window_size // 2
else:                     # 大圖
    adaptive_target_drop = target_drop
    adaptive_eta_beta = eta_beta
    adaptive_window_size = window_size
```

### 主退火循環

```python
for each trial:
    # 重置監控變數
    energy_window.clear()
    current_alpha = 0.8
    best_energy = float('inf')
    last_improve_cycle = 0
    
    I0 = I0_min  # 初始溫度
    cycle = 0
    
    while I0 <= I0_max:
        for i in range(tau):
            # 1. 標準pSA退火步驟
            if config == 5:
                # 動態調整lambda參數
                if len(energy_window) > 5:
                    adaptive_factor = 1.0 + (current_alpha - 0.8) * 0.5
                    adaptive_lambda = lambda_var * adaptive_factor
                    cuda.memcpy_htod(lambda_gpu, adaptive_lambda)
                
                # 執行GPU kernel
                annealing_kernel(vertex, I0, h_vector_gpu, J_matrix_gpu, 
                               spin_vector_gpu, rnd_ini_gpu, lambda_gpu, 
                               delta_gpu, nu_gpu, count_device)
            
            # 2. 自適應監控（每10步執行一次）
            if i % 10 == 0:
                # 從GPU複製當前解
                cuda.memcpy_dtoh(temp_spin, spin_vector_gpu)
                current_energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)
                energy_window.append(current_energy)
                
                # 更新最佳能量
                if current_energy < best_energy:
                    best_energy = current_energy
                    last_improve_cycle = cycle
        
        # 3. 溫度更新
        I0 /= current_beta  # 使用自適應調整的beta值
        cycle += 1
```

### 自適應調整機制

#### Alpha調整（搜索強度控制）
```python
if len(energy_window) >= 20:
    recent_improvement = energy_window[-20] - energy_window[-1]
    if recent_improvement < adaptive_target_drop * 0.5:
        current_alpha = min(0.9, current_alpha + 0.05)  # 增加搜索強度
    elif recent_improvement > adaptive_target_drop * 2:
        current_alpha = max(0.1, current_alpha - 0.05)  # 減少搜索強度
```

#### Beta調整（退火速度控制）
```python
if len(energy_window) == window_size:
    delta = energy_window[0] - energy_window[-1]
    current_beta += adaptive_eta_beta * (adaptive_target_drop - delta)
    current_beta = max(beta * 0.5, min(beta * 2.0, current_beta))
```

#### 停滯檢測與恢復
```python
if (cycle - last_improve_cycle) > stall_threshold:
    print(f"[Recovery] Stagnation at cycle {cycle}, performing cluster-flip")
    temp_spin = cluster_flip(temp_spin, J_matrix, flip_probability=0.3)
    cuda.memcpy_htod(spin_vector_gpu, temp_spin)
    last_improve_cycle = cycle
```

---

## 參數詳解

### 核心自適應參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|---------|------|
| `eta_alpha` | float | 1e-3 | Alpha調整學習率，控制搜索強度的變化速度 |
| `eta_beta` | float | 1e-3 | Beta調整學習率，控制退火速度的變化速度 |
| `target_drop` | float | 100 | 期望的能量下降幅度，用於評估收斂進度 |
| `window_size` | int | 100 | 能量監控窗口大小，影響調整的敏感度 |
| `stall_threshold` | int | 200 | 停滯檢測閾值，超過此週期數觸發恢復機制 |

### 基礎pSA參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `l_scale` | float | Lambda參數的標準差，控制設備變異性 |
| `d_scale` | float | Delta參數的標準差，控制偏移變異性 |
| `n_scale` | float | Nu參數的標準差，控制更新頻率變異性 |
| `tau` | int | 每個溫度下的迭代次數 |
| `res` | int | 時間解析度，影響更新頻率 |

### 圖結構自適應縮放

```python
# 小圖 (≤50 頂點)：需要更精細的調整
adaptive_target_drop = target_drop * 0.05
adaptive_eta_beta = eta_beta * 2.0
adaptive_window_size = window_size // 4

# 中圖 (50-200 頂點)：中等調整
adaptive_target_drop = target_drop * 0.3
adaptive_eta_beta = eta_beta * 1.2  
adaptive_window_size = window_size // 2

# 大圖 (>200 頂點)：使用原始參數
adaptive_target_drop = target_drop
adaptive_eta_beta = eta_beta
adaptive_window_size = window_size
```

---

## 使用方法

### 基本調用
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

### 參數調優建議

#### 保守策略（推薦用於初始測試）
```bash
--eta_alpha 0.0 \
--eta_beta 0.0005 \
--target_drop 50 \
--window_size 150 \
--stall_threshold 300
```

#### 中等策略
```bash
--eta_alpha 0.0005 \
--eta_beta 0.001 \
--target_drop 80 \
--window_size 100 \
--stall_threshold 200
```

#### 激進策略（需謹慎使用）
```bash
--eta_alpha 0.001 \
--eta_beta 0.002 \
--target_drop 100 \
--window_size 70 \
--stall_threshold 100
```

---

## 性能分析

### 理論優勢

1. **自適應收斂**：根據實際收斂狀況調整策略
2. **圖結構感知**：針對不同圖特性優化參數
3. **停滯恢復**：自動檢測並處理收斂停滯
4. **參數魯棒性**：減少對初始參數選擇的依賴

### 實際表現

#### 成功案例（G55圖，5000頂點）
- **收斂模式**：穩定下降至負值範圍
- **最終能量**：-7000 ~ -8000
- **可達性**：85-90%

#### 挑戰案例（G1圖，800頂點，密度3.64%）
- **問題**：能量振盪，收斂不穩定
- **原因**：密集圖的複雜能量景觀 + 頻繁的GPU同步
- **能量範圍**：-246 到 +2044（不穩定）

### 與其他演算法比較

| 演算法 | 能量穩定性 | 可達性 | 收斂速度 | 參數敏感度 |
|--------|------------|---------|----------|------------|
| **pSA (Config 2)** | 中等 | 76.9% | 中等 | 高 |
| **SpSA (Config 4)** | 優秀 | 98.6% | 快 | 低 |
| **ApSA (Config 5)** | 不穩定 | 79.7% | 變化大 | 極高 |

---

## 已知問題與解決方案

### 主要問題

#### 1. GPU同步衝突
**問題**：在tau循環內部頻繁進行GPU↔CPU記憶體同步
```python
# 有問題的代碼
if config in [5] and i % 10 == 0:
    cuda.memcpy_dtoh(temp_spin, spin_vector_gpu)  # 頻繁同步
    current_energy = energy_calculate(...)
```

**影響**：破壞GPU並行效率，導致性能下降

#### 2. 參數干擾
**問題**：在退火過程中動態修改核心參數
```python
# 可能導致不穩定的調整
adaptive_lambda = lambda_var * adaptive_factor
cuda.memcpy_htod(lambda_gpu, adaptive_lambda)
```

**影響**：干擾退火過程的自然收斂

#### 3. 監控頻率過高
**問題**：每10步就進行能量計算和參數調整

**影響**：計算開銷大，可能導致振盪

### 解決方案

#### 保守自適應策略
```python
# 減少調整頻率
monitoring_frequency = 50  # 從10改為50

# 降低調整幅度  
eta_alpha = 0.0      # 禁用alpha調整
eta_beta = 0.0005    # 大幅降低beta調整

# 增加穩定性參數
window_size = 150    # 增大窗口
stall_threshold = 300 # 增大停滯閾值
```

#### 分階段調整
```python
# 第一階段：純pSA運行（0-300週期）
# 第二階段：開始溫和自適應（300-700週期）  
# 第三階段：正常自適應（700週期以後）
```

---

## 最佳實踐

### 參數選擇指南

#### 對於小圖（<100頂點）
```bash
--eta_alpha 0.0 \
--eta_beta 0.0005 \
--target_drop 20 \
--window_size 200 \
--stall_threshold 400
```

#### 對於中圖（100-1000頂點）
```bash
--eta_alpha 0.0005 \
--eta_beta 0.001 \
--target_drop 50 \
--window_size 100 \
--stall_threshold 250
```

#### 對於大圖（>1000頂點）
```bash
--eta_alpha 0.001 \
--eta_beta 0.002 \
--target_drop 100 \
--window_size 70 \
--stall_threshold 200
```

### 調試建議

1. **先測試穩定性**：從eta_alpha=0.0, eta_beta=0.0開始
2. **逐步增加自適應**：慢慢增加eta_beta值
3. **監控能量曲線**：確保能量為負值且穩定下降
4. **比較基準性能**：與Config 2 (pSA)和Config 4 (SpSA)比較

### 性能優化

1. **減少同步頻率**：將監控頻率從每10步改為每20-50步
2. **批量調整**：將多個小調整合併為較少的大調整
3. **條件觸發**：只在特定條件下才執行自適應調整

---

## 未來改進方向

### 短期改進

1. **修復同步問題**：將自適應邏輯移到tau循環外部
2. **簡化參數**：減少需要調整的參數數量
3. **提升穩定性**：使用更保守的調整策略

### 長期發展

1. **機器學習指導**：使用ML預測最佳參數組合
2. **多層自適應**：實現層次化的自適應機制
3. **實時性能監控**：開發更精確的性能評估指標

---

## 結論

ApSA (Config 5) 代表了在pSA基礎上加入自適應機制的有益嘗試。雖然當前實現在穩定性方面存在挑戰，但其核心理念——根據實際收斂狀況動態調整參數——是正確的方向。

**建議使用場景**：
- 研究和實驗環境
- 對特定圖結構進行參數探索
- 作為其他自適應演算法的參考實現

**不建議使用場景**：
- 生產環境的關鍵任務
- 需要穩定可預測結果的應用
- 對計算效率要求極高的場景

通過持續的改進和優化，ApSA有潛力成為一個強大且實用的自適應退火演算法。 