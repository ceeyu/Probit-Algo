import math
import numpy as np
import random
import time
import statistics
import csv
import os
import sys
import argparse
import matplotlib.pyplot as plt
import datetime
import pandas as pd

"""
========================================
硬體層級模擬 - Crossbar 權重轉換
========================================

本程式碼模擬硬體 Crossbar 陣列的實際限制：

1. 硬體限制：
   - Crossbar 的權重只能存 {0, 0.5, 1}（電壓/電導範圍限制）
   - Spin 暫存器存 {0, 1}（或在某些版本使用 {-1, 1}）
   
2. 演算法需求：
   - GSET 問題的權重是 {-1, 0, 1}
   - Ising 模型需要有正負號的耦合
   
3. 解決方案 - 數位修正電路：
   
   (A) 權重轉換：
       理想 J ∈ {-1, 0, 1}
       硬體 J_hw = (J + 1) / 2 ∈ {0, 0.5, 1}
   
   (B) Spin 轉換（僅用於 MVM 計算）：
       理想 Ising spin: s ∈ {-1, 1}
       Binary spin: b = (s + 1) / 2 ∈ {0, 1}
   
   (C) MVM 修正公式：
       硬體計算：I_hw = J_hw @ b
       數位修正：I = 4*I_hw - 2*J_hw_row_sums - 2*b_sum + N
       
       其中：
       - I 是理想的本地場（演算法需要的）
       - I_hw 是硬體 Crossbar 計算的結果
       - J_hw_row_sums[i] = Σ_j J_hw[i,j]
       - b_sum = Σ_j b_j（對 binary spin）或 m_sum = Σ_j m_j（對 Ising spin）
       - N 是節點數
   
   推導過程：
       設 J_hw = (J+1)/2, b = (s+1)/2
       則 I_hw = Σ_j J_hw_ij * b_j 
               = Σ_j ((J_ij+1)/2) * ((s_j+1)/2)
               = (1/4) * Σ_j (J_ij*s_j + J_ij + s_j + 1)
               = (1/4) * [I + Σ_j J_ij + Σ_j s_j + N]
       因此 I = 4*I_hw - Σ_j J_ij - Σ_j s_j - N
              = 4*I_hw - 2*J_hw_row_sums + N - s_sum - N
              = 4*I_hw - 2*J_hw_row_sums - s_sum
       
       對於 binary spin: s = 2b - 1, s_sum = 2*b_sum - N
       代入：I = 4*I_hw - 2*J_hw_row_sums - (2*b_sum - N)
              = 4*I_hw - 2*J_hw_row_sums - 2*b_sum + N

4. 重要原則：
   - 能量計算（calculate_energy）和 Cut 值計算（calculate_cut）永遠使用原始的 J_matrix 和 G_matrix
   - 只有在 MVM（矩陣向量乘法）計算時才使用 J_hw 和修正公式
   - GSET 的 best-known cut 值不會因為硬體限制而改變（問題本身不變）
   - 硬體轉換是透明的：從演算法的角度看，本地場 I 的值與理想情況完全一致

5. 修改的函數：
   - probit_annealing_synchronous: Binary spin 同步版本（RPA）
   - probit_fitting_hardware_synchronous: Binary spin 同步版本（機率遮罩）
   - probit_annealing: Ising spin 非同步版本（MCS）
   - traditional_sa: Ising spin Metropolis-Hastings

所有函數都已加入硬體 Crossbar MVM 模擬與數位修正電路。

6. 硬體雜訊產生器（取代 Gaussian noise）：
   
   架構：Bitmask Generator + 2n MUX + 2n TRNG + All-1 Xbar
   
   ┌─────────────────────────────────────────────────────────────┐
   │  k ──→ [Bitmask Generator] ──→ bitmask (k 個 1)             │
   │                  │                                           │
   │          ┌───────┴───────┐                                   │
   │          ↓               ↓                                   │
   │    ┌─────────┐     ┌─────────┐                               │
   │    │ MUX 組1 │     │ MUX 組2 │                               │
   │    │ 0→0     │     │ 0→1     │  ← 2n 個類比 TRNG 輸入        │
   │    │ 1→TRNG  │     │ 1→TRNG  │                               │
   │    └────┬────┘     └────┬────┘                               │
   │         ↓               ↓                                    │
   │    ┌─────────────────────────┐                               │
   │    │     All-1 Xbar          │                               │
   │    │   (2n 輸入加總)          │                               │
   │    └───────────┬─────────────┘                               │
   │                ↓                                             │
   │         total_output                                         │
   │                │                                             │
   │         noise = total_output - n (數位修正)                   │
   └─────────────────────────────────────────────────────────────┘
   
   數學原理（中央極限定理近似高斯分佈）：
   - 前 n 個 MUX: (n-k) 個輸出 0，k 個輸出 TRNG
   - 後 n 個 MUX: (n-k) 個輸出 1，k 個輸出 TRNG
   - total = Σ(k 個 TRNG_group1) + (n-k) + Σ(k 個 TRNG_group2)
   - E[total] = k/2 + (n-k) + k/2 = n
   - noise = total - n → E[noise] = 0, Var[noise] = k/2, σ = √(k/2)
   
   溫度控制：
   - k=n (全1): 最大隨機性（高溫）
   - k=0 (全0): 無隨機性（低溫）
   - 退火過程中 k 從 n 遞減到 0
========================================
"""

def hardware_bitmask_noise_generator(n, k):
    """
    硬體 Bitmask + MUX + TRNG + All-1 Xbar 雜訊產生器
    
    此函數模擬以下硬體電路來產生類高斯雜訊（使用中央極限定理）：
    
    硬體元件：
    1. Bitmask Generator: 根據 k 產生 n-bit 遮罩
       - k=0: 0000...00
       - k=1: 0000...01
       - k=2: 0000...11
       - k=n: 1111...11
    
    2. 2n 個類比 TRNG: 產生 {0, 1} 的隨機位元
    
    3. 2n 個 MUX (多工器):
       - 前 n 個 MUX: bitmask[i]=0 → 輸出 0, bitmask[i]=1 → 輸出 TRNG_i
       - 後 n 個 MUX: bitmask[i]=0 → 輸出 1, bitmask[i]=1 → 輸出 TRNG_{n+i}
    
    4. All-1 Xbar: 矩陣元素全為 1，將 2n 個 MUX 輸出加總
    
    數學分析：
    - 當 bitmask 有 k 個 1 時：
      - MUX 組1 輸出: (n-k) 個 0 + k 個 TRNG = Σ_{i=1}^{k} TRNG_i
      - MUX 組2 輸出: (n-k) 個 1 + k 個 TRNG = (n-k) + Σ_{i=n+1}^{n+k} TRNG_i
      - total = Σ_{i=1}^{k} TRNG_i + (n-k) + Σ_{i=n+1}^{n+k} TRNG_i
      - E[total] = k/2 + (n-k) + k/2 = n
      - Var[total] = k/4 + k/4 = k/2
    
    - 數位修正: noise = total - n
      - E[noise] = 0 (均值為 0)
      - σ_noise = √(k/2) (標準差隨 k 增加)
    
    參數:
        n: spin 數量（也是每組 MUX/TRNG 的數量）
        k: bitmask 中 1 的數量 (0 <= k <= n)，控制隨機性強度
           k=0: 無隨機性（低溫，確定性輸出 0）
           k=n: 最大隨機性（高溫，σ = √(n/2)）
    
    返回:
        noise: 標量雜訊值（int 類型，均值為 0，標準差為 √(k/2)）
    """
    # === 步驟 1: Bitmask Generator ===
    # 產生 k 個 1，從低位開始：[1,1,...,1,0,0,...,0]
    #                        |-- k 個 --|-- n-k 個 -|
    bitmask = np.zeros(n, dtype=np.int32)
    if k > 0:
        bitmask[:k] = 1
    
    # === 步驟 2: 2n 個類比 TRNG (產生 0 或 1，各 50% 機率) ===
    trngs_group1 = np.random.randint(0, 2, size=n)  # TRNG_1 到 TRNG_n
    trngs_group2 = np.random.randint(0, 2, size=n)  # TRNG_{n+1} 到 TRNG_{2n}
    
    # === 步驟 3: 2n 個 MUX ===
    # 前 n 個 MUX: bitmask[i]=0 → 輸出 0, bitmask[i]=1 → 輸出 TRNG_i
    mux_output_group1 = np.where(bitmask == 1, trngs_group1, 0)
    
    # 後 n 個 MUX: bitmask[i]=0 → 輸出 1, bitmask[i]=1 → 輸出 TRNG_i
    mux_output_group2 = np.where(bitmask == 1, trngs_group2, 1)
    
    # === 步驟 4: All-1 Xbar (將所有 2n 個輸入加總) ===
    total_output = np.sum(mux_output_group1) + np.sum(mux_output_group2)
    
    # === 步驟 5: 數位修正（減去期望值使均值為 0）===
    # E[total_output] = k/2 + (n-k) + k/2 = n
    noise = total_output - n
    
    return noise


def sigma_to_bitmask_k(sigma, sigma_start, sigma_end, n):
    """
    將 sigma（退火溫度）映射到 bitmask 的 k 值
    
    ========================================
    正確映射公式（修正版）
    ========================================
    
    硬體雜訊標準差公式: σ_hw = √(k/2)
    反推 k 的公式: k = 2 × σ²
    
    這樣可以讓硬體雜訊強度與軟體 Gaussian noise 匹配：
    - sigma = 5.0  → k = 2 × 25 = 50  → σ_hw = √(50/2) = 5.0 ✓
    - sigma = 1.0  → k = 2 × 1  = 2   → σ_hw = √(2/2)  = 1.0 ✓
    - sigma = 0.01 → k = 2 × 0.0001 ≈ 0 → σ_hw ≈ 0 ✓
    
    硬體簡化優點：
    - 當 sigma_start = 5.0 時，k_max = 50（而非 N=800）
    - 大幅減少所需的 TRNG 通道數量
    - k 的範圍從 [0, N] 縮小到 [0, 50]
    
    參數:
        sigma: 當前的 sigma 值
        sigma_start: 起始 sigma（高溫）
        sigma_end: 結束 sigma（低溫）
        n: spin 數量（用於上限保護，但通常不會達到）
    
    返回:
        k: bitmask 中 1 的數量
    """
    # 正確映射: k = 2 × sigma²
    k = int(round(2.0 * sigma * sigma)) 
    
    # 確保 k 在合理範圍內 [0, n]
    k = max(0, min(n, k))
    
    return k


def parse_arguments():
    parser = argparse.ArgumentParser(description="Probit Annealing vs Traditional SA Comparison")
    parser.add_argument('--file_path', type=str, required=True, help="File path for GSET graph data")
    parser.add_argument('--trial', type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument('--timesteps', type=int, default=10000, help="Number of annealing timesteps (default: 10000)")
    parser.add_argument('--sigma_start', type=float, default=5.0, help="Starting sigma for Probit (default: 5.0)")
    parser.add_argument('--sigma_end', type=float, default=0.01, help="Ending sigma for Probit (default: 0.01)")
    parser.add_argument('--T_start', type=float, default=5.0, help="Starting temperature for Traditional SA (default: 5.0)")
    parser.add_argument('--T_end', type=float, default=0.01, help="Ending temperature for Traditional SA (default: 0.01)")
    parser.add_argument('--schedule', type=str, default='linear', choices=['exponential', 'linear'], 
                       help="Annealing schedule type (default: exponential)")
    parser.add_argument('--probit_mode', type=str, default='synchronous', choices=['synchronous', 'asynchronous'],
                       help="Probit update mode: synchronous (parallel hardware) or asynchronous (MCS) (default: synchronous)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="RPA ratio: fraction of spins allowed to update per timestep (synchronous mode) (default: 0.1)")
    return parser.parse_args()

def read_file_MAXCUT(file_path):
    """讀取 GSET 格式的圖檔案"""
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()   # 節點數
        second_line = f.readline().strip()  # 邊數
        third_line = f.readline().strip()   # 邊值類型
        fourth_line = f.readline().strip()  # best-known value
        lines = f.readlines()
    return first_line, second_line, third_line, fourth_line, lines

def get_graph_MAXCUT(vertex, lines):
    """
    從邊列表建構鄰接矩陣
    
    注意：這裡返回的是**原始的** GSET 權重 {-1, 0, 1}
    硬體轉換會在演算法內部處理
    """
    G_matrix = np.zeros((vertex, vertex), dtype=np.int32)
    
    line_count = len(lines)
    print(f'節點數: {vertex}, 邊數: {line_count}')
    
    for line_text in lines:
        weight_list = list(map(int, line_text.split(' ')))
        i = weight_list[0] - 1  # GSET 從 1 開始編號，把node對應到0開始的index
        j = weight_list[1] - 1  # GSET 從 1 開始編號，把node對應到0開始的index
        
        # 保持原始權重 {-1, 0, 1}
        G_matrix[i, j] = weight_list[2]
    
    # 對稱化
    G_matrix = G_matrix + G_matrix.T
    
    print(f'[圖形權重] 範圍 [{np.min(G_matrix)}, {np.max(G_matrix)}]')
    
    return G_matrix

def calculate_energy(spin_vector_input, J_matrix):
    """
    通用 Spin 能量計算（自動檢測 binary {0,1} 或 Ising {-1,+1}）
    
    轉換關係：s = 2b - 1（從 binary {0,1} 轉到 Ising {-1,+1}）
    然後使用標準 Ising 能量公式計算
    
    能量公式（與 gpu_MAXCUT_var0708.py 一致）：
    h_vector = diag(J_matrix)
    energy = -(J_energy + h_energy)
    其中 J_energy = ((J m - h) dot m) / 2, h_energy = (h dot m)
    """
    N = J_matrix.shape[0]
    
    # 自動檢測輸入類型並轉換為 Ising spin {-1,+1}
    # 如果是binary spin {0,1}，則轉換為 Ising spin {-1,+1}來計算能量
    if np.all((spin_vector_input == 0) | (spin_vector_input == 1)):
        # Binary spin {0,1} → Ising spin {-1,+1}
        s_vector = 2 * spin_vector_input - 1
    elif np.all((spin_vector_input == -1) | (spin_vector_input == 1)):
        # 已經是 Ising spin {-1,+1}
        s_vector = spin_vector_input
    else:
        # 容許浮點數誤差
        s_vector = 2 * spin_vector_input - 1
    
    spin_vector = np.reshape(s_vector, (N, 1)) #reshape成N*1的矩陣，列向量(N行、1列)
    h_vector = np.reshape(np.diag(J_matrix), (N, 1))# 對角線元素
    h_energy = np.sum(h_vector * spin_vector)# 對角線元素的能量
    J_energy = np.sum((np.dot(J_matrix, spin_vector) - h_vector) * spin_vector) / 2# 非對角線元素
    return -(J_energy + h_energy)

def calculate_cut(spin_vector_input, G_matrix):
    """
    通用 Cut 值計算（自動檢測 binary {0,1} 或 Ising {-1,+1}）
    
    轉換關係：s = 2b - 1（從 binary {0,1} 轉到 Ising {-1,+1}）
    然後使用標準公式計算
    
    Cut 值公式（與 gpu_MAXCUT_var0708.py 一致）：
    對於每條邊 (i,j)，如果兩端 spin 不同則計入
    (1 - s_i*s_j) 當不同號時 = 2，相同號時 = 0
    """
    N = len(spin_vector_input)
    
    # 自動檢測輸入類型並轉換為 Ising spin {-1,+1}
    if np.all((spin_vector_input == 0) | (spin_vector_input == 1)):
        # Binary spin {0,1} → Ising spin {-1,+1}
        s_vector = 2 * spin_vector_input - 1
    elif np.all((spin_vector_input == -1) | (spin_vector_input == 1)):
        # 已經是 Ising spin {-1,+1}
        s_vector = spin_vector_input
    else:
        # 容許浮點數誤差
        s_vector = 2 * spin_vector_input - 1
    
    spin = np.reshape(s_vector, (N,)) # spin是一維陣列 [s1,s2, ... ,sn]
    upper_triangle = np.triu_indices(N, k=1) #上三角，從主對角線上方位移一個位置
    cut_val = np.sum(G_matrix[upper_triangle] * (1 - np.outer(spin, spin)[upper_triangle]))
    return cut_val / 2  # 除以 2 是因為 (1 - s_i*s_j) 對不同號給出 2

def probit_fitting_hardware_synchronous(J_matrix, timesteps, sigma_start, sigma_end, schedule='linear', record_energy=False, epsilon=0.1):
    """
    Probit 類比退火法 - Binary Spin 硬體版本（b ∈ {0, 1}）
    RPA 策略（Ratio-Controlled Parallel Annealing - Amorphica 論文）
    
    硬體模擬限制:
    - J_hw (Crossbar) 存 {0, 0.5, 1}
    - b (Spin 暫存器) 存 {0, 1}
    
    每個 timestep：
    1. Crossbar 平行 MVM: I_hw = J_hw × b（硬體類比計算）
    2. 數位修正電路: I = 4*I_hw - 2*J_hw_row_sums - 2*b_sum + N（還原理想本地場）
    3. 平行 TRNG: 產生 noise_vector（全局相同雜訊）
    4. 平行決策: b_proposed = (I + noise > 0) ? 1 : 0
    5. RPA 機率遮罩: 使用 LFSR 決定哪些 spin 可以更新
    
    參數:
        epsilon: RPA 比例，控制每個 timestep 更新的 spin 比例 (default: 0.1 = 10%)
    """
    N = J_matrix.shape[0] # 如果 J 是一個100*100的矩陣，則J_matrix.shape會回傳(100,100)
    
    # === 硬體矩陣預計算 ===
    # 1. 建立硬體權重矩陣 J_hw
    # 理想 J ∈ {-1, 0, 1}, J_hw = (J + 1) / 2 ∈ {0, 0.5, 1}
    J_hw_matrix = (J_matrix + 1) / 2.0
    
    # 2. 預先計算 J_hw 的每行加總（用於 MVM 修正公式）
    J_hw_row_sums = np.sum(J_hw_matrix, axis=1)
    
    print(f'[硬體模擬] J 範圍: [{np.min(J_matrix):.1f}, {np.max(J_matrix):.1f}] → J_hw 範圍: [{np.min(J_hw_matrix):.1f}, {np.max(J_hw_matrix):.1f}]')
    
    # 1. 初始化為 Binary Spin {0, 1}
    b_vector = np.random.choice([0, 1], size=N).astype(np.float64)
    
    # 2. 定義退火排程
    if schedule == 'exponential':
        alpha = (sigma_end / sigma_start) ** (1.0 / timesteps)
        annealing_schedule = sigma_start * (alpha ** np.arange(timesteps))
    else: 
        annealing_schedule = np.linspace(sigma_start, sigma_end, timesteps)
    
    energy_history = []
    
    # 記錄初始能量（使用理想的 J_matrix）
    if record_energy:
        current_energy = calculate_energy(b_vector, J_matrix) # b_vector 是 binary spin {0,1}，所以會被calculate_energy轉為{-1,1}
        energy_history.append(current_energy)
    
    # === Bitmask Generator 監控設定 ===
    # 計算新公式下的 k 範圍 (k = 2 × sigma²)
    # 讓 k從 2*5*5 到 2*0.01*0.01
    k_start = int(round(2.0 * sigma_start * sigma_start))
    k_end = int(round(2.0 * sigma_end * sigma_end))
    
    # print(f'\n[Bitmask Generator 監控] - 使用正確映射公式 k = 2σ²')
    print(f'  n (spin 數量) = {N}')
    print(f'  sigma 範圍: {sigma_start} → {sigma_end}')
    print(f'  k 範圍 (k = 2σ²): {k_start} → {k_end} (僅需 {k_start} 個 TRNG 通道)')
    print(f'  理論 σ_noise 範圍: √({k_start}/2)={np.sqrt(k_start/2):.2f} → √({k_end}/2)={np.sqrt(k_end/2):.2f}')
    
    # 3. RPA 同步迴圈（真平行 + 部分更新）
    for t in range(timesteps):
        sigma = annealing_schedule[t]
        
        # === 步驟 1: 真平行 MVM（硬體 Crossbar 類比計算 + 數位修正）===
        # (A) 硬體 Crossbar MVM: J_hw {0, 0.5, 1} @ b {0, 1}
        I_hw_vector = np.dot(J_hw_matrix, b_vector)
        
        # (B) 數位修正電路: 還原出理想的本地場 I = J @ s = J @ (2b-1)
        # 推導: I_hw = J_hw @ b = ((J+1)/2) @ b
        #       I = 4*I_hw - 2*J_hw_row_sums - 2*b_sum + N
        b_sum = np.sum(b_vector) #用硬體參數結果還原比較值
        I_vector = 4.0 * I_hw_vector - 2.0 * J_hw_row_sums - 2.0 * b_sum + N
        
        # === 步驟 2: 硬體雜訊產生（Bitmask + MUX + TRNG + All-1 Xbar）===
        # 硬體架構：
        #   - Bitmask Generator: 根據當前溫度(sigma)產生遮罩
        #   - 2k 個 MUX 啟用: 前 k 個 (1→TRNG)，後 k 個 (1→TRNG)
        #   - 2k 個類比 TRNG: 產生 {0, 1}
        #   - All-1 Xbar: 將所有輸出加總
        #   - 數位修正: noise = total - N (使均值為 0)
        #
        # 正確映射公式: k = 2 × sigma²
        # 這樣 σ_hw = √(k/2) = √(2σ²/2) = σ（與軟體匹配！）
        
        k = sigma_to_bitmask_k(sigma, sigma_start, sigma_end, N)
        common_noise = hardware_bitmask_noise_generator(N, k)
        noise_vector = np.full(N, float(common_noise))
        '''
        # === Bitmask Generator 監控輸出（每個 timestep 都輸出）===
        # 產生完整 bitmask
        full_bitmask = '1' * k + '0' * (N - k)
        sigma_theory = np.sqrt(k / 2) if k > 0 else 0
        
        print(f'\n{"="*80}')
        print(f'Timestep: {t} | sigma: {sigma:.4f} | k: {k} | k/n: {k/N:.2%} | noise: {common_noise} | σ_theory: {sigma_theory:.2f}')
        print(f'{"="*80}')
        print(f'完整 Bitmask (n={N}, k={k}個1):')
        
        # 每行顯示 64 位元
        line_width = 64
        for i in range(0, N, line_width):
            line = full_bitmask[i:i+line_width]
            # 標註位置
            print(f'  [{i:4d}-{min(i+line_width-1, N-1):4d}]: {line}')
        '''
        # 步驟 3: Binary 決策（threshold = 0）
        # 如果 I + noise > 0，則 b = 1；否則 b = 0
        decision_signal = I_vector + noise_vector
        b_vector_proposed = (decision_signal > 0).astype(np.float64)

        # === 步驟 4: RPA 機率遮罩 (硬體 LFSR 實現) ===
        # 硬體實現對應：
        #   - LFSR_1...LFSR_n: 每個 spin 產生一個 uniform [0,1) 隨機數
        #   - 比較器: 檢查 LFSR_i 是否 < V_rpa (epsilon)
        #   - 邏輯: LFSR_i < epsilon → 允許更新 spin_i
        #   - 註：epsilon = 0.1 → 10% 更新率
        
        update_probability = np.random.rand(N)  # LFSR
        update_mask = update_probability < epsilon  # 比較器
        
        # 最終決策邏輯閘：只更新被允許的 spin，其餘保持原值
        b_vector_new = np.where(update_mask, b_vector_proposed, b_vector)
        
        # 步驟 5: 更新
        b_vector = b_vector_new
        
        # 記錄能量
        if record_energy:
            current_energy = calculate_energy(b_vector, J_matrix)
            energy_history.append(current_energy)
    
    if record_energy:
        return b_vector, energy_history
    else:
        return b_vector, None

def traditional_sa(J_matrix, timesteps, T_start, T_end, schedule='linear', record_energy=False):
    """
    傳統模擬退火法 (Metropolis-Hastings)
    
    參數:
        J_matrix: Ising 耦合矩陣
        timesteps: 退火步數
        T_start: 起始溫度
        T_end: 結束溫度
        schedule: 'exponential' 或 'linear'
        record_energy: 是否記錄每步能量
    
    返回:
        m_vector: 最終自旋配置
        energy_history: 能量歷史
    """
    N = J_matrix.shape[0]
    
    # 1. 初始化
    m_vector = np.random.choice([-1, 1], size=N)
    current_energy = calculate_energy(m_vector, J_matrix)
    
    # 2. 定義退火排程
    if schedule == 'exponential':
        alpha = (T_end / T_start) ** (1.0 / timesteps)
        annealing_schedule = T_start * (alpha ** np.arange(timesteps))
    else:  # linear
        annealing_schedule = np.linspace(T_start, T_end, timesteps)
    
    energy_history = [] if record_energy else None
    
    # 3. Metropolis 迴圈
    for t in range(timesteps):
        T = annealing_schedule[t]
        
        # 隨機選擇一個自旋翻轉
        i = np.random.randint(0, N)
        
        # 計算能量變化 ΔE
        # ΔE = E_new - E_old = 2 * m_i * (sum_j J_ij * m_j)
        local_field = np.dot(J_matrix[i, :], m_vector)
        delta_E = 2 * m_vector[i] * local_field
        
        # Metropolis 接受準則
        if delta_E <= 0:
            # 能量下降，接受
            m_vector[i] *= -1
            current_energy += delta_E
        else:
            # 能量上升，以機率接受
            if T > 1e-10 and np.random.random() < np.exp(-delta_E / T):
                m_vector[i] *= -1
                current_energy += delta_E
        
        # 記錄能量
        if record_energy:
            energy_history.append(current_energy)
    
    if record_energy:
        return m_vector, energy_history
    else:
        return m_vector, None

def run_comparison_experiment(args, J_matrix, G_matrix, best_known):
    """
    執行 Probit vs Traditional SA 的比較實驗
    """
    N = J_matrix.shape[0]
    
    print("\n" + "="*80)
    print("開始比較實驗：Probit Annealing vs Traditional SA")
    print("="*80)
    print(f"節點數 (N): {N}")
    print(f"試驗次數: {args.trial}")
    print(f"退火步數: {args.timesteps}")
    print(f"退火排程: {args.schedule}")
    print(f"Probit 模式: {args.probit_mode}")
    print(f"Probit 參數: sigma_start={args.sigma_start}, sigma_end={args.sigma_end}")
    if args.probit_mode == 'synchronous':
        print(f"  → RPA 同步模式（Ratio-Controlled Parallel Annealing - Amorphica 論文）")
        print(f"  → 平行計算：每個 timestep 平行計算所有 {N} 個 I_vector (硬體 Crossbar)")
        print(f"  → 部分更新：每個 timestep 最多更新 {int(N * args.epsilon)} 個 spin (epsilon={args.epsilon})")
    else:
        print(f"  → 非同步更新：每個 timestep 更新 {N} 個 spin (MCS)")
    print(f"Traditional SA 參數: T_start={args.T_start}, T_end={args.T_end}")
    print(f"  → 每個 timestep 更新 1 個 spin")
    print("="*80 + "\n")
    
    # 儲存結果
    probit_energies = []
    probit_cuts = []
    probit_times = []
    
    sa_energies = []
    sa_cuts = []
    sa_times = []
    
    # 最後一次試驗記錄完整能量曲線
    probit_energy_history = None
    sa_energy_history = None
    
    # 執行試驗
    for trial in range(args.trial):
        print(f"\n===== Trial {trial + 1}/{args.trial} =====")
        
        # === Probit Annealing ===
        record_last = (trial == args.trial - 1)
        
        t_start = time.perf_counter()
        if args.probit_mode == 'synchronous':
            m_probit, energy_hist = probit_fitting_hardware_synchronous( #要使用的演算法，可以更改
                J_matrix, args.timesteps, args.sigma_start, args.sigma_end, 
                args.schedule, record_energy=record_last, epsilon=args.epsilon
            )
            ''' 
            原本的asynchronous 演算法 Monte Carlo
        else:  # asynchronous
            m_probit, energy_hist = probit_annealing( # 使用 Monte Carlo
                J_matrix, args.timesteps, args.sigma_start, args.sigma_end, 
                args.schedule, record_energy=record_last
            )
            '''
        t_end = time.perf_counter()
        probit_time = (t_end - t_start) * 1000  # ms
        
        probit_energy = calculate_energy(m_probit, J_matrix)
        probit_cut = calculate_cut(m_probit, G_matrix)
        
        probit_energies.append(probit_energy)
        probit_cuts.append(probit_cut)
        probit_times.append(probit_time)
        
        if record_last:
            probit_energy_history = energy_hist
        
        print(f"[Probit] Energy: {probit_energy:.2f}, Cut: {probit_cut}, Time: {probit_time:.2f} ms")
        
        # === Traditional SA ===
        t_start = time.perf_counter()
        m_sa, energy_hist_sa = traditional_sa(
            J_matrix, args.timesteps, args.T_start, args.T_end, 
            args.schedule, record_energy=record_last
        )
        t_end = time.perf_counter()
        sa_time = (t_end - t_start) * 1000  # ms
        
        sa_energy = calculate_energy(m_sa, J_matrix)
        sa_cut = calculate_cut(m_sa, G_matrix)
        
        sa_energies.append(sa_energy)
        sa_cuts.append(sa_cut)
        sa_times.append(sa_time)
        
        if record_last:
            sa_energy_history = energy_hist_sa
        
        print(f"[Traditional SA] Energy: {sa_energy:.2f}, Cut: {sa_cut}, Time: {sa_time:.2f} ms")
    
    # 統計分析
    print("\n" + "="*80)
    print("統計結果")
    print("="*80)
    
    print("\n--- Probit Annealing ---")
    print(f"能量: 平均={np.mean(probit_energies):.2f}, 標準差={np.std(probit_energies):.2f}, 最小={np.min(probit_energies):.2f}, 最大={np.max(probit_energies):.2f}")
    print(f"Cut值: 平均={np.mean(probit_cuts):.2f}, 標準差={np.std(probit_cuts):.2f}, 最小={np.min(probit_cuts)}, 最大={np.max(probit_cuts)}")
    print(f"時間: 平均={np.mean(probit_times):.2f} ms")
    print(f"達成率: 平均={100*np.mean(probit_cuts)/best_known:.2f}%, 最大={100*np.max(probit_cuts)/best_known:.2f}%")
    
    print("\n--- Traditional SA ---")
    print(f"能量: 平均={np.mean(sa_energies):.2f}, 標準差={np.std(sa_energies):.2f}, 最小={np.min(sa_energies):.2f}, 最大={np.max(sa_energies):.2f}")
    print(f"Cut值: 平均={np.mean(sa_cuts):.2f}, 標準差={np.std(sa_cuts):.2f}, 最小={np.min(sa_cuts)}, 最大={np.max(sa_cuts)}")
    print(f"時間: 平均={np.mean(sa_times):.2f} ms")
    print(f"達成率: 平均={100*np.mean(sa_cuts)/best_known:.2f}%, 最大={100*np.max(sa_cuts)/best_known:.2f}%")
    
    print("\n--- 比較 ---")
    energy_diff = np.mean(probit_energies) - np.mean(sa_energies)
    cut_diff = np.mean(probit_cuts) - np.mean(sa_cuts)
    print(f"能量差異 (Probit - SA): {energy_diff:.2f}")
    print(f"Cut值差異 (Probit - SA): {cut_diff:.2f}")
    
    return {
        'probit_energies': probit_energies,
        'probit_cuts': probit_cuts,
        'probit_times': probit_times,
        'probit_energy_history': probit_energy_history,
        'sa_energies': sa_energies,
        'sa_cuts': sa_cuts,
        'sa_times': sa_times,
        'sa_energy_history': sa_energy_history
    }

def save_results_and_plots(args, results, file_base, best_known):
    """儲存結果到 CSV 和圖表"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 檢查是否有環境變數指定輸出目錄（用於批次執行）
    output_dir = os.environ.get('HARDWARE_OUTPUT_DIR', './hw_noise_probit_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    mode_suffix = 'sync' if args.probit_mode == 'synchronous' else 'async'
    
    # === 1. 儲存統計結果到 CSV ===
    csv_filename = f'{output_dir}/comparison_{file_base}_{mode_suffix}_trial{args.trial}_steps{args.timesteps}_{timestamp}.csv'
    
    summary_data = {
        'Algorithm': ['Probit', 'Traditional SA'],
        'Mean Energy': [np.mean(results['probit_energies']), np.mean(results['sa_energies'])],
        'Std Energy': [np.std(results['probit_energies']), np.std(results['sa_energies'])],
        'Min Energy': [np.min(results['probit_energies']), np.min(results['sa_energies'])],
        'Max Energy': [np.max(results['probit_energies']), np.max(results['sa_energies'])],
        'Mean Cut': [np.mean(results['probit_cuts']), np.mean(results['sa_cuts'])],
        'Std Cut': [np.std(results['probit_cuts']), np.std(results['sa_cuts'])],
        'Min Cut': [np.min(results['probit_cuts']), np.min(results['sa_cuts'])],
        'Max Cut': [np.max(results['probit_cuts']), np.max(results['sa_cuts'])],
        'Mean Time (ms)': [np.mean(results['probit_times']), np.mean(results['sa_times'])],
        'Mean Accuracy (%)': [100*np.mean(results['probit_cuts'])/best_known, 100*np.mean(results['sa_cuts'])/best_known],
        'Max Accuracy (%)': [100*np.max(results['probit_cuts'])/best_known, 100*np.max(results['sa_cuts'])/best_known]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(csv_filename, index=False)
    print(f"\n統計結果已儲存: {csv_filename}")
    
    # === 2. 繪製能量分佈直方圖 ===
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    probit_label = f'Probit ({args.probit_mode})'
    plt.hist(results['probit_energies'], bins=20, alpha=0.7, label=probit_label, color='blue', edgecolor='black')
    plt.hist(results['sa_energies'], bins=20, alpha=0.7, label='Traditional SA', color='red', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title(f'能量分佈直方圖 ({file_base}, {args.trial} trials)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(results['probit_cuts'], bins=20, alpha=0.7, label=probit_label, color='blue', edgecolor='black')
    plt.hist(results['sa_cuts'], bins=20, alpha=0.7, label='Traditional SA', color='red', edgecolor='black')
    plt.axvline(best_known, color='green', linestyle='--', linewidth=2, label=f'Best Known ({best_known})')
    plt.xlabel('Cut Value')
    plt.ylabel('Frequency')
    plt.title(f'Cut值分佈直方圖 ({file_base}, {args.trial} trials)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_filename = f'{output_dir}/histogram_{file_base}_{mode_suffix}_trial{args.trial}_{timestamp}.png'
    plt.savefig(hist_filename, dpi=150)
    plt.close()
    print(f"能量分佈圖已儲存: {hist_filename}")
    
    # === 3. 繪製能量演化曲線（最後一次試驗）===
    if results['probit_energy_history'] is not None and results['sa_energy_history'] is not None:
        plt.figure(figsize=(12, 6))
        
        probit_label = f'Probit ({args.probit_mode})'
        plt.plot(results['probit_energy_history'], label=probit_label, color='blue', linewidth=1.5, alpha=0.8)
        plt.plot(results['sa_energy_history'], label='Traditional SA', color='red', linewidth=1.5, alpha=0.8)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Energy')
        plt.title(f'能量演化曲線 ({file_base}, Last Trial)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        evolution_filename = f'{output_dir}/energy_evolution_{file_base}_{mode_suffix}_steps{args.timesteps}_{timestamp}.png'
        plt.savefig(evolution_filename, dpi=150)
        plt.close()
        print(f"能量演化曲線已儲存: {evolution_filename}")
    
    # === 4. 儲存詳細結果到 Excel ===
    excel_filename = f'{output_dir}/detailed_results_{file_base}_{mode_suffix}_trial{args.trial}_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 統計摘要
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # 詳細結果
        df_details = pd.DataFrame({
            'Trial': range(1, args.trial + 1),
            'Probit_Energy': results['probit_energies'],
            'Probit_Cut': results['probit_cuts'],
            'Probit_Time_ms': results['probit_times'],
            'SA_Energy': results['sa_energies'],
            'SA_Cut': results['sa_cuts'],
            'SA_Time_ms': results['sa_times']
        })
        df_details.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # 能量演化（最後一次試驗）
        if results['probit_energy_history'] is not None:
            # 確保兩個列表長度相同（Probit 多記錄了初始能量）
            min_len = min(len(results['probit_energy_history']), len(results['sa_energy_history']))
            df_evolution = pd.DataFrame({
                'Timestep': range(min_len),
                'Probit_Energy': results['probit_energy_history'][:min_len],
                'SA_Energy': results['sa_energy_history'][:min_len]
            })
            df_evolution.to_excel(writer, sheet_name='Energy_Evolution', index=False)
    
    print(f"詳細結果已儲存: {excel_filename}")

def main():
    args = parse_arguments()
    
    print("="*80)
    print("Probit 類比退火法 vs 傳統模擬退火法")
    print("="*80)
    
    # 讀取 GSET 問題
    first_line, second_line, third_line, fourth_line, lines = read_file_MAXCUT(args.file_path)
    vertex = int(first_line)
    best_known = int(fourth_line)
    
    file_path = args.file_path
    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    
    print(f"\n圖形: {file_base}")
    print(f"節點數: {vertex}")
    print(f"Best-known Cut: {best_known}")
    
    # 建構矩陣
    G_matrix = get_graph_MAXCUT(vertex, lines)
    J_matrix = -G_matrix  # For MAXCUT: J = -G，轉換矩陣
    
    # 執行比較實驗
    results = run_comparison_experiment(args, J_matrix, G_matrix, best_known)
    
    # 儲存結果
    save_results_and_plots(args, results, file_base, best_known)
    
    print("\n" + "="*80)
    print("實驗完成！")
    print("="*80)

if __name__ == "__main__":
    main()

