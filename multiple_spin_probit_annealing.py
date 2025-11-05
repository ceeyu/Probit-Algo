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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Probit Annealing vs Traditional SA Comparison")
    parser.add_argument('--file_path', type=str, required=True, help="File path for GSET graph data")
    parser.add_argument('--trial', type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument('--timesteps', type=int, default=10000, help="Number of annealing timesteps (default: 10000)")
    parser.add_argument('--sigma_start', type=float, default=5.0, help="Starting sigma for Probit (default: 5.0)")
    parser.add_argument('--sigma_end', type=float, default=0.01, help="Ending sigma for Probit (default: 0.01)")
    parser.add_argument('--T_start', type=float, default=5.0, help="Starting temperature for Traditional SA (default: 5.0)")
    parser.add_argument('--T_end', type=float, default=0.01, help="Ending temperature for Traditional SA (default: 0.01)")
    parser.add_argument('--schedule', type=str, default='exponential', choices=['exponential', 'linear'], 
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
    """從邊列表建構鄰接矩陣"""
    G_matrix = np.zeros((vertex, vertex), dtype=np.int32)
    
    line_count = len(lines)
    print(f'節點數: {vertex}, 邊數: {line_count}')
    
    for line_text in lines:
        weight_list = list(map(int, line_text.split(' ')))
        i = weight_list[0] - 1  # GSET 從 1 開始編號，把node對應到0開始的index
        j = weight_list[1] - 1  # GSET 從 1 開始編號，把node對應到0開始的index
        G_matrix[i, j] = weight_list[2]
    
    # 對稱化
    G_matrix = G_matrix + G_matrix.T
    return G_matrix

def calculate_energy(m_vector, J_matrix):
    """
    與 gpu_MAXCUT_var0708.py 一致的能量計算方式：
    h_vector = diag(J_matrix)
    energy = -(J_energy + h_energy)
    其中 J_energy = ((J m - h) dot m) / 2, h_energy = (h dot m)
    """
    N = J_matrix.shape[0]
    spin_vector = np.reshape(m_vector, (N, 1))
    h_vector = np.reshape(np.diag(J_matrix), (N, 1)) # 對角線元素
    h_energy = np.sum(h_vector * spin_vector) # 對角線元素的能量
    J_energy = np.sum((np.dot(J_matrix, spin_vector) - h_vector) * spin_vector) / 2 # 非對角線元素
    return -(J_energy + h_energy)

def calculate_cut(m_vector, G_matrix):
    """
    與 gpu_MAXCUT_var0708.py 一致的 Cut 值計算方式（向量化，上三角）：
    使用 cut_calculate 之向量化邏輯。
    """
    N = len(m_vector)
    spin = np.reshape(m_vector, (N,))
    upper_triangle = np.triu_indices(N, k=1)
    # 計算 cut 值：對於上三角的每條邊，如果兩端 spin 不同則計入
    # (1 - spin[i]*spin[j]) 當不同號時 = (1-(-1)) = 2，相同號時 = (1-1) = 0
    cut_val = np.sum(G_matrix[upper_triangle] * (1 - np.outer(spin, spin)[upper_triangle]))
    return int(cut_val / 2)  # 除以 2 是因為 (1 - spin[i]*spin[j]) 對不同號給出 2

def probit_annealing_synchronous(J_matrix, timesteps, sigma_start, sigma_end, schedule='linear', record_energy=False, epsilon=0.1):
    """
    Probit 類比退火法（RPA 策略 - Ratio-Controlled Parallel Annealing - Amorphica 論文）
    
    每個 timestep：
    1. Crossbar 平行 MVM: I = J × m（一次計算所有 I_vector）
    2. 平行 TRNG: 產生 noise_vector
    3. 平行決策: m_proposed = sgn(I + noise)（計算所有提議的新狀態）
    4. RPA 遮罩: 只允許 N * epsilon 個 spin 實際更新（Policy-Dependent Masker）
    
    參數:
        epsilon: RPA 比例，控制每個 timestep 更新的 spin 比例 (default: 0.1 = 10%)
    """
    N = J_matrix.shape[0]
    
    # 1. 初始化
    m_vector = np.random.choice([-1, 1], size=N).astype(np.float64)
    
    # 2. 定義退火排程
    if schedule == 'exponential':
        alpha = (sigma_end / sigma_start) ** (1.0 / timesteps)
        annealing_schedule = sigma_start * (alpha ** np.arange(timesteps))
    else: 
        annealing_schedule = np.linspace(sigma_start, sigma_end, timesteps)
    
    energy_history = []
    
    # 記錄初始能量
    if record_energy:
        current_energy = calculate_energy(m_vector, J_matrix)
        energy_history.append(current_energy)
    
    # 3. RPA 同步迴圈（真平行 + 部分更新）
    for t in range(timesteps):
        sigma = annealing_schedule[t]
        
        # 步驟 1: 真平行 MVM (Crossbar 一次算出所有 I)
        I_vector = np.dot(J_matrix, m_vector)
        
        # 步驟 2: 真平行 TRNG (一次產生所有雜訊)
        # noise_vector = np.random.normal(0.0, sigma, size=N)
        common_noise = np.random.normal(0.0, sigma)
        noise_vector = np.full(N, common_noise)
        
        # 步驟 3: 真平行決策 (計算「提議」的新狀態)
        m_vector_proposed = np.sign(I_vector + noise_vector)
        
        # 處理 sgn(0) 的情況：保持原值
        zero_mask = (m_vector_proposed == 0)
        m_vector_proposed[zero_mask] = m_vector[zero_mask]

        # === 步驟 4: RPA 遮罩 (Amorphica 論文的核心 - Policy-Dependent Masker) ===
        # 找出所有「想要翻轉」的 spin (即 m_proposed != m_vector)
        flippable_indices = np.where(m_vector_proposed != m_vector)[0] # 想要翻轉的spin的index
        num_flippable = len(flippable_indices) # 想要翻轉的spin的數量
        
        # 計算我們「允許」翻轉多少個
        num_allowed_flips = int(N * epsilon)
        
        # 建立一個全為「保持原值」的新向量
        m_vector_new = m_vector.copy()

        if num_flippable > 0:
            if num_flippable <= num_allowed_flips:
                # 想翻轉的 spin 數量 <= 允許數量，全部翻轉
                m_vector_new[flippable_indices] = m_vector_proposed[flippable_indices]
            else:
                # 想翻轉的 spin 數量 > 允許數量，隨機挑選 num_allowed_flips 個來翻轉
                indices_to_flip = np.random.choice(flippable_indices, size=num_allowed_flips, replace=False)
                m_vector_new[indices_to_flip] = m_vector_proposed[indices_to_flip]
        
        # 步驟 5: 更新
        m_vector = m_vector_new
        
        # 記錄能量
        if record_energy:
            current_energy = calculate_energy(m_vector, J_matrix)
            energy_history.append(current_energy)
    
    if record_energy:
        return m_vector, energy_history
    else:
        return m_vector, None

def probit_annealing(J_matrix, timesteps, sigma_start, sigma_end, schedule='linear', record_energy=False):
    """
    Probit 類比退火法（非同步更新 - Monte Carlo Sweep）
    
    每個 timestep 執行 N 次隨機單一 spin 更新
    """
    N = J_matrix.shape[0]
    
    # 1. 初始化
    m_vector = np.random.choice([-1, 1], size=N)
    
    # 2. 定義退火排程
    if schedule == 'exponential':
        alpha = (sigma_end / sigma_start) ** (1.0 / timesteps)
        annealing_schedule = sigma_start * (alpha ** np.arange(timesteps))
    else: 
        annealing_schedule = np.linspace(sigma_start, sigma_end, timesteps)
    
    energy_history = []
    
    # 如果要記錄能量，我們需要追蹤當前能量以加快速度
    current_energy = calculate_energy(m_vector, J_matrix)
    if record_energy:
        energy_history.append(current_energy)
    
    # 3. 非同步 Probit 迴圈 (Monte Carlo Sweep)
    for t in range(timesteps):
        sigma = annealing_schedule[t]
        
        # 每個 timestep 執行 N 次更新（一個完整的 MCS）
        for _ in range(N):
            # 步驟 1: 隨機選擇一個自旋
            i = np.random.randint(0, N)
            
            # 步驟 2: MVM (只計算第 i 個自旋的本地場)
            # I_i = sum_j(J_ij * m_j)
            I_i = np.dot(J_matrix[i, :], m_vector)
            
            # 步驟 3: TRNG (只取一個雜訊)
            noise_i = np.random.normal(0.0, sigma)
            
            # 步驟 4: 拔河比賽
            # m_i_new = sgn(I_i + noise_i)
            new_m_i = np.sign(I_i + noise_i)
            
            if new_m_i == 0:
                new_m_i = 1 # 處理 sgn(0) 的情況
            
            # 步驟 5: 立刻更新自旋
            if m_vector[i] != new_m_i:
                old_m_i = m_vector[i]
                m_vector[i] = new_m_i
                
                # 更新能量 (使用 Delta E，更有效率)
                # delta_E_physics = E_new - E_old = -2 * m_i_old * I_i
                delta_E = 2 * old_m_i * I_i
                current_energy += delta_E
        
        # 每個 MCS 記錄一次能量
        if record_energy:
            energy_history.append(current_energy)
    
    if record_energy:
        return m_vector, energy_history
    else:
        # 如果不記錄，我們需要返回最終計算的m_vector
        return m_vector, None
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
            m_probit, energy_hist = probit_annealing_synchronous(
                J_matrix, args.timesteps, args.sigma_start, args.sigma_end, 
                args.schedule, record_energy=record_last, epsilon=args.epsilon
            )
        else:  # asynchronous
            m_probit, energy_hist = probit_annealing(
                J_matrix, args.timesteps, args.sigma_start, args.sigma_end, 
                args.schedule, record_energy=record_last
            )
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
    output_dir = './multiple_spin_probit_comparison_results'
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
    J_matrix = -G_matrix  # For MAXCUT: J = -G
    
    # 執行比較實驗
    results = run_comparison_experiment(args, J_matrix, G_matrix, best_known)
    
    # 儲存結果
    save_results_and_plots(args, results, file_base, best_known)
    
    print("\n" + "="*80)
    print("實驗完成！")
    print("="*80)

if __name__ == "__main__":
    main()

