import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import random
import time
import statistics
import csv
import os
import sys
import argparse
import cProfile
import pstats
import io
from collections import deque
import ast
import matplotlib.pyplot as plt
import datetime
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify GPU device number")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device number (default: 0)")
    parser.add_argument('--file_path', type=str, required=True, help="File path for data")
    parser.add_argument('--param', type=int, default=1, help="Parameter type (default: 1)")
    parser.add_argument('--cycle', type=int, default=1000, help="Number of cycles (default: 1000)")
    parser.add_argument('--trial', type=int, default=100, help="Number of trials (default: 100)")
    parser.add_argument('--tau', type=int, default=1, help="tau (default: 1)")
    parser.add_argument('--res', type=int, default=10, help="res (default: 1)")
    parser.add_argument('--thread', type=int, default=32, help="Number of threads (default: 32)")
    parser.add_argument('--config', type=int, default=2, help="Configuration (default: 2)")
    parser.add_argument('--unique', type=int, default=1, help="Unique noise magnitude (default: 1)")
    parser.add_argument('--mean_range', type=int, default=4, help="Configuration (default: 1)")
    parser.add_argument('--stall_prop', type=float, default=0.5, help='stalled prop值')
    parser.add_argument('--l_scale', type=float, default=0, help='lambda_sigma')
    parser.add_argument('--d_scale', type=float, default=0, help='delta_sigma')
    parser.add_argument('--n_scale', type=float, default=0, help='nu_sigma')
    
    # === 新增的自適應和混合演算法參數 ===
    parser.add_argument('--cooling_stages', type=str, 
        default='[(0.0,0.3),(0.3,0.7),(0.7,1.0)]',
        help='Multi-stage cooling schedule as list of (start_frac,end_frac)')
    parser.add_argument('--stall_threshold', type=int, default=200,
        help='Cycles without improvement before triggering cluster-flip')
    parser.add_argument('--window_size', type=int, default=100,
        help='Window size for adaptive temperature scheduling')
    parser.add_argument('--eta_alpha', type=float, default=1e-3,
        help='Learning rate for alpha adjustment')
    parser.add_argument('--eta_beta', type=float, default=1e-3,
        help='Learning rate for beta adjustment')
    parser.add_argument('--target_drop', type=float, default=100,
        help='Target energy drop per window for adaptive scheduling')
    parser.add_argument('--num_populations', type=int, default=3,
        help='Number of populations for multi-population algorithm')
    parser.add_argument('--migrate_interval', type=int, default=50,
        help='Migration interval for multi-population')
    parser.add_argument('--migrate_size', type=float, default=0.1,
        help='Migration size as fraction of population')
    
    return parser.parse_args()

def initialize_cuda(args):
    cuda.Device(args.gpu).make_context()
    cuda.init()
    device = cuda.Device(args.gpu)
    properties = device.get_attributes()
    max_shared_mem = properties[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
    print(f"Maximum shared memory size: {max_shared_mem} bytes")
    return device

def load_gpu_code(config):
    if config == 1:
        cu_file = 'ssa_annealing_kernel.cu'
    elif config == 2:
        cu_file = 'psa_annealing_kernel_var.cu'
    elif config == 3:
        cu_file = 'tapsa_annealing_kernel_var.cu'
    elif config == 4:
        cu_file = 'spsa_annealing_kernel_var.cu'
    # === 新增的混合和自適應演算法 ===
    elif config == 5:
        cu_file = 'apsa_annealing_kernel_var.cu'  # Adaptive PSA - 自適應溫度調度的pSA
    elif config == 6:
        cu_file = 'hpsa_annealing_kernel_var.cu'  # Hybrid PSA - 混合多演算法框架
    elif config == 7:
        cu_file = 'mpsa_annealing_kernel_var.cu'  # Multi-Population PSA - 多族群pSA
    elif config == 8:
        cu_file = 'smart_annealing_kernel_var.cu' # Smart Adaptive PSA - 智能自適應演算法
    # === 新增：對應 gpu_MAXCUT_var0702.py 的演算法 ===
    elif config == 9:
        cu_file = 'tapsa_annealing_kernel_var0702.cu'  # TApSA mean range 10~1
    elif config == 10:
        cu_file = 'spsa_annealing_kernel_var0702.cu'  # SpSA 偶數bit stall prop=0.5, 奇數直接更新
    else:
        raise ValueError(f"Unsupported config: {config}")

    with open(cu_file, 'r') as file:
        gpu_code = file.read()
    return gpu_code

def read_file_MAXCUT(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        third_line = f.readline().strip()
        fourth_line = f.readline().strip()
        lines = f.readlines()
    return first_line, second_line, third_line, fourth_line, lines

# Function to calculate the number of cuts
def cut_calculate(vertex, G_matrix, spin_vector):
    spin_vector_reshaped = np.reshape(spin_vector, (len(spin_vector),)) # Convert spin_vector to 1D array
    upper_triangle = np.triu_indices(len(spin_vector), k=1) # Get the indices of the upper triangular matrix
    cut_val = np.sum(G_matrix[upper_triangle] * (1 - np.outer(spin_vector_reshaped, spin_vector_reshaped)[upper_triangle])) # Calculate cut_val by multiplying the upper triangular elements with the corresponding entries in G_matrix
    return int(cut_val / 2)

# Function to calculate the energy
def energy_calculate(vertex, h_vector, J_matrix, spin_vector):
    h_energy = np.sum(h_vector * spin_vector)
    J_energy = np.sum((np.dot(J_matrix, spin_vector) - h_vector) * spin_vector) / 2
    return -(J_energy + h_energy)

def get_graph_MAXCUT(vertex, lines):
    G_matrix = np.zeros((vertex, vertex), int)
    
    # Count the number of lines (edges)
    line_count = len(lines)
    print('Number of Edges:', line_count)
    
    # Iterate through the lines to construct the adjacency matrix
    for line_text in lines:
        weight_list = list(map(int, line_text.split(' ')))  # Convert space-separated string to list of integers
        i = weight_list[0] - 1
        j = weight_list[1] - 1
        G_matrix[i, j] = weight_list[2]  # Assign weight to the corresponding entry in the matrix
    
    # Adding the matrix to its transpose to make it symmetric
    G_matrix = G_matrix + G_matrix.T
    return G_matrix

def set_annealing_parameters(vertex, args, h_vector, J_matrix):
    mean_each = []
    std_each = []
    for j in range(vertex):
        mean_each.append((vertex-1) * np.mean(J_matrix[j]))
        std_each.append(np.sqrt((vertex-1) * np.var(np.concatenate([J_matrix[j], -J_matrix[j]]))))
    sigma = np.mean(std_each)
    mean = np.mean(mean_each)

    min_cycle = np.int32(args.cycle)
    trial = np.int32(args.trial)
    tau = np.int32(args.tau)
    Mshot = np.int32(1)

    sigma_vector = np.array(std_each, dtype=np.float32).reshape((-1, 1))
    unique = args.unique
    if unique == 0:
        nrnd_vector = np.float32(0.67448975 * np.mean(sigma_vector) * np.ones((vertex, 1)))
    elif unique == 1:
        nrnd_vector = np.float32(0.67448975 * sigma_vector)
    nrnd_max = np.max(nrnd_vector)

    param = args.param
    if param == 1:
        I0_min = np.float32(np.max(sigma_vector) * 0.01 + np.min(np.abs(mean_each)))
        I0_max = np.float32(np.max(sigma_vector) * 2 + np.min(np.abs(mean_each)))
    elif param == 2: 
        I0_min = np.float32(0.1/sigma)
        I0_max = np.float32(10/sigma)

    beta = np.float32((I0_min / I0_max) ** (tau / (min_cycle - 1)))
    max_cycle = math.ceil((math.log10(I0_min / I0_max) / math.log10(beta))) * tau

    threads_per_block = (1, args.thread, 1)
    block_x = math.ceil(1 / threads_per_block[0])
    block_y = math.ceil(vertex / threads_per_block[1])
    blocks_per_grid = (block_x, block_y, 1)

    return min_cycle, trial, tau, Mshot, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid

def cleanup_cuda():
    cuda.Context.pop()

def check_cuda_error(message=""):
    try:
        cuda.Context.synchronize()  # 同期を行い、エラーチェックをする
    except pycuda._driver.Error as e:
        raise RuntimeError(f"CUDA error during {message}: {str(e)}")

def adaptive_temperature_schedule(cycle, max_cycles, I0_min, I0_max, stages):
    """
    多階段自適應溫度調度
    stages: list of (start_frac, end_frac) tuples
    """
    frac = cycle / max_cycles
    
    for i, (start_frac, end_frac) in enumerate(stages):
        if start_frac <= frac < end_frac:
            # 每個階段內的進度
            stage_progress = (frac - start_frac) / (end_frac - start_frac)
            
            # 不同階段使用不同的冷卻曲線
            if i == 0:  # 第一階段：快速冷卻
                return I0_max * (I0_min/I0_max)**(stage_progress**0.5)
            elif i == 1:  # 第二階段：線性冷卻
                return I0_max * (I0_min/I0_max)**stage_progress
            else:  # 最後階段：緩慢冷卻
                return I0_max * (I0_min/I0_max)**(stage_progress**2)
    
    return I0_min

def cluster_flip(spin_vector, J_matrix, flip_probability=0.3):
    """
    條件式Cluster-Flip機制：當停滯時執行群集翻轉
    """
    vertex = len(spin_vector)
    new_spin = spin_vector.copy()
    
    # 找出能量最高的節點作為cluster seed
    energies = []
    for i in range(vertex):
        local_energy = np.sum(J_matrix[i] * spin_vector.flatten()) * spin_vector[i]
        energies.append(local_energy)
    
    # 選擇前20%高能量節點進行cluster flip
    high_energy_indices = np.argsort(energies)[-int(vertex * 0.2):]
    
    for idx in high_energy_indices:
        if random.random() < flip_probability:
            new_spin[idx] *= -1
            # 擴散到鄰居節點
            neighbors = np.where(J_matrix[idx] != 0)[0]
            for neighbor in neighbors:
                if random.random() < flip_probability * 0.5:
                    new_spin[neighbor] *= -1
    
    return new_spin

def run_trials(args, file_base, config, vertex, min_cycle, trial, tau, Mshot, gpu_code, h_vector, G_matrix, J_matrix, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid):
    # Add compiler flags (same as original)
    nvcc_flags = ["-std=c++14", "--compiler-options", "-fno-strict-aliasing"]
    # Load the module with the modified compilation command
    mod = SourceModule(gpu_code, options=nvcc_flags, no_extern_c=True)
    annealing_kernel = mod.get_function("annealing_module")
    cut_calculate_kernel = mod.get_function("calculate_cut_val")
    
    # === 新增：Config 5 GPU自適應功能所需的額外kernels ===
    if config == 5:
        init_adaptive_state_kernel = mod.get_function("init_adaptive_state")
        get_adaptive_params_kernel = mod.get_function("get_adaptive_params")
        init_curand_states_kernel = mod.get_function("init_curand_states")
        simple_annealing_kernel = mod.get_function("simple_annealing_module")  # 使用專門的簡化版kernel

    stall_prop = args.stall_prop
    mean_range = args.mean_range
    l_scale = args.l_scale
    d_scale = args.d_scale
    n_scale = args.n_scale
    res = args.res

    # === 根據圖大小調整參數 ===
    if vertex <= 50:  # 小圖 - 需要更激進的調整
        adaptive_target_drop = max(args.target_drop * 0.05, 1)  # 極小的目標下降
        adaptive_eta_beta = args.eta_beta * 2.0         # 更積極的調整
        adaptive_window_size = min(args.window_size // 4, 20)  # 更小的視窗
        print(f"Small graph detected (V={vertex}): target_drop={adaptive_target_drop:.1f}, eta_beta={adaptive_eta_beta:.6f}, window_size={adaptive_window_size}")
    elif vertex <= 200:  # 中圖
        adaptive_target_drop = args.target_drop * 0.3
        adaptive_eta_beta = args.eta_beta * 1.2
        adaptive_window_size = args.window_size // 2
        print(f"Medium graph detected (V={vertex}): target_drop={adaptive_target_drop:.1f}, eta_beta={adaptive_eta_beta:.6f}, window_size={adaptive_window_size}")
    else:  # 大圖
        adaptive_target_drop = args.target_drop
        adaptive_eta_beta = args.eta_beta
        adaptive_window_size = args.window_size
        print(f"Large graph detected (V={vertex}): target_drop={adaptive_target_drop:.1f}, eta_beta={adaptive_eta_beta:.6f}, window_size={adaptive_window_size}")

    # === 新增：Adaptive 變數初始化 ===
    from collections import deque
    best_energy        = float('inf')
    last_improve_cycle = 0
    energy_window      = deque(maxlen=adaptive_window_size)
    current_beta       = beta

    # === 新增：自適應和混合演算法參數 ===
    cooling_stages = ast.literal_eval(args.cooling_stages)
    stall_threshold = args.stall_threshold
    window_size = adaptive_window_size
    eta_alpha = args.eta_alpha
    eta_beta = args.eta_beta
    target_drop = adaptive_target_drop
    num_populations = args.num_populations
    migrate_interval = args.migrate_interval
    migrate_size = args.migrate_size

    print('Number of trials:', trial)
    print("Min Cycles:", min_cycle)
    print('beta:', beta)
    print('I0_min:', I0_min)
    print('I0_max:', I0_max)
    print('tau:', tau)
    print('res:', res)
    print(f'Config: {config}, Adaptive features enabled')
    print(f'Cooling stages: {cooling_stages}')
    print(f'Stall threshold: {stall_threshold}')

    if config == 1:
        nrnd_vector_gpu = cuda.mem_alloc(nrnd_vector.nbytes)
        cuda.memcpy_htod(nrnd_vector_gpu, nrnd_vector)

    h_vector_int8 = h_vector.astype(np.int8)
    h_vector_gpu = cuda.mem_alloc(h_vector_int8.nbytes)
    cuda.memcpy_htod(h_vector_gpu, h_vector_int8)

    J_matrix_int8 = J_matrix.astype(np.int8)
    J_matrix_gpu = cuda.mem_alloc(J_matrix_int8.nbytes)
    cuda.memcpy_htod(J_matrix_gpu, J_matrix_int8)

    time_list = [] 
    cut_list = []
    energy_list = []
    energy_per_cycle_last_trial = [] # 記錄最後一個 trial 的每個 cycle 能量
    
    for k in range(trial + 1):
        print("######## Trial", k + 1, " ###########")

        spin_vector = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.int32)
        spin_vector_gpu = cuda.mem_alloc(spin_vector.nbytes)
        cuda.memcpy_htod(spin_vector_gpu, spin_vector)

        # === 新增：Config 5 GPU自適應狀態初始化 ===
        if config == 5:
            # 分配GPU記憶體給自適應狀態結構
            adaptive_state_size = 4 * (4 + 4 + 4 + 4 + 4 + 100 + 4 + 4 + 4)  # 估算結構大小
            adaptive_state_gpu = cuda.mem_alloc(adaptive_state_size)
            
            # 分配隨機數生成器狀態
            rand_states_gpu = cuda.mem_alloc(vertex * 48)  # curandState約為48bytes
            
            # 初始化自適應狀態
            init_adaptive_state_kernel(adaptive_state_gpu, np.int32(adaptive_window_size),
                                     block=(1, 1, 1), grid=(1, 1, 1))
            
            # 初始化隨機數生成器
            block_size = min(256, vertex)
            grid_size = (vertex + block_size - 1) // block_size
            init_curand_states_kernel(rand_states_gpu, np.uint64(int(time.time())), np.int32(vertex),
                                    block=(block_size, 1, 1), grid=(grid_size, 1, 1))
            
            print(f"GPU adaptive state initialized for Config 5")

        # === 新增：自適應監控變數初始化 ===
        best_energy = float('inf')
        last_improve_cycle = 0
        energy_window = deque(maxlen=window_size)
        current_alpha = 0.8  # 初始擾動強度
        current_beta = beta
        algorithm_performance = {'pSA': 0, 'TApSA': 0, 'SpSA': 0}
        current_algorithm = 'pSA'  # 當前使用的演算法
        
        # 多族群初始化（config 7 使用）
        if config == 7:
            populations = []
            population_fitness = []  # 追蹤各族群的適應度
            for pop_idx in range(num_populations):
                pop_spin = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.int32)
                pop_spin_gpu = cuda.mem_alloc(pop_spin.nbytes)
                cuda.memcpy_htod(pop_spin_gpu, pop_spin)
                populations.append(pop_spin_gpu)
                population_fitness.append(float('inf'))  # 初始適應度為無窮大

        if config == 1:
            rnd_ini = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.float32)
        elif config in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
            lambda_var = np.random.normal(loc=1.0, scale=l_scale, size=(vertex, 1)).astype(np.float32)
            lambda_var = np.maximum(lambda_var, 1e-9)
            lambda_gpu = cuda.mem_alloc(lambda_var.nbytes)
            cuda.memcpy_htod(lambda_gpu, lambda_var)
            delta =  np.random.normal(loc=0, scale=d_scale, size=(vertex, 1)).astype(np.float32)
            delta_gpu = cuda.mem_alloc(delta.nbytes)
            cuda.memcpy_htod(delta_gpu, delta)
            nu = np.random.normal(loc=res, scale=n_scale, size=(vertex, 1)).astype(np.int32)
            nu = np.maximum(nu, 1)
            nu_gpu = cuda.mem_alloc(nu.nbytes)
            cuda.memcpy_htod(nu_gpu, nu)
            count_device = 0

        rnd_ini_gpu = cuda.mem_alloc(rnd_ini.nbytes)

        if config == 1:
            Itanh_ini = np.zeros((vertex, 1), dtype=np.float32)
            Itanh_ini_gpu = cuda.mem_alloc(Itanh_ini.nbytes)
            cuda.memcpy_htod(Itanh_ini_gpu, Itanh_ini)
        elif config == 3:
            D_res = np.zeros((vertex*mean_range, 1), dtype=np.float32)
            D_res_gpu = cuda.mem_alloc(D_res.nbytes)
            cuda.memcpy_htod(D_res_gpu, D_res)
            shared_mem_size = 4 * threads_per_block[1] * mean_range
        elif config == 9:  # TApSA 0702版本
            D_res = np.zeros((vertex*mean_range, 1), dtype=np.float32)
            D_res_gpu = cuda.mem_alloc(D_res.nbytes)
            cuda.memcpy_htod(D_res_gpu, D_res)
            shared_mem_size = 4 * threads_per_block[1] * mean_range
            
        cut_val = np.zeros(1, dtype=np.float32)
        cut_val_gpu = cuda.mem_alloc(cut_val.nbytes)
        cuda.memcpy_htod(cut_val_gpu, cut_val)

        time_start_gpu = cuda.Event()
        time_end_gpu = cuda.Event()
        time_start_gpu.record()
        
        # === 修改：主退火迴圈加入自適應機制 ===
        cycle = 0
        I0 = I0_min
        
        # === 新增：config 9 的動態 mean_range 參數 ===
        if config == 9:
            I0_max_log = np.log10(I0_max)
            I0_min_log = np.log10(I0_min)
            mean_range_min = 1 #主要設定mean range
            mean_range_max = 10
        
        while I0 <= I0_max:
            # 1. 自適應溫度調度（只有 config 8 使用）
            if config == 8:
                I0 = adaptive_temperature_schedule(cycle, min_cycle, I0_min, I0_max, cooling_stages)
            
            # === 新增：config 9 的動態 mean_range 調整 ===
            if config == 9:
                I0_log = np.log10(I0)
                # 線性插值調整 mean_range
                dynamic_mean_range = int(round(mean_range_min + (mean_range_max - mean_range_min) * (I0_log - I0_min_log) / (I0_max_log - I0_min_log)))
                dynamic_mean_range = max(mean_range_min, min(dynamic_mean_range, mean_range_max))
                shared_mem_size = 4 * threads_per_block[1] * dynamic_mean_range
            
            for i in range(tau):
                # 2. 執行退火步驟
                if config == 1:
                    rnd_ini = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, Itanh_ini_gpu, rnd_ini_gpu, nrnd_vector_gpu,
                                    block=threads_per_block, grid=blocks_per_grid)
                elif config == 2:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                    block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                elif config == 3:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    annealing_kernel(np.int32(mean_range), np.int32(vertex), I0,  h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu, D_res_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                    block=threads_per_block, grid=blocks_per_grid, shared=shared_mem_size)
                    count_device = count_device + 1
                elif config == 4:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    annealing_kernel(np.float32(stall_prop), np.int32(vertex), I0,  h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                    block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                # === 新增演算法 ===
                elif config == 5:  # Adaptive PSA - 優化的GPU端自適應實現
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    
                    # === 簡化的GPU端自適應實現 ===
                    # 統一使用標準annealing_kernel，GPU端處理自適應邏輯
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, 
                                   spin_vector_gpu, rnd_ini_gpu, lambda_gpu, delta_gpu, nu_gpu, 
                                   np.int32(count_device),
                                   # 自適應參數
                                   adaptive_state_gpu,
                                   np.float32(adaptive_target_drop), 
                                   np.float32(eta_alpha), 
                                   np.float32(adaptive_eta_beta),
                                   np.int32(args.stall_threshold), 
                                   np.int32(cycle),
                                   rand_states_gpu,
                                   block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                elif config == 6:  # Hybrid PSA - 動態演算法切換
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    
                    # 根據當前最佳演算法動態調整參數
                    if current_algorithm == 'pSA':
                        # 標準 pSA 參數
                        hybrid_lambda = lambda_var
                        hybrid_nu = nu
                    elif current_algorithm == 'TApSA':
                        # 模擬 TApSA：增強 lambda，調整 nu 做時間平均
                        hybrid_lambda = lambda_var * 1.5
                        hybrid_nu = np.maximum(nu // 2, 1)  # 更頻繁更新
                    else:  # SpSA
                        # 模擬 SpSA：降低擾動，基於停滯調整
                        stall_factor = max(0.5, 1.0 - (cycle - last_improve_cycle) / stall_threshold)
                        hybrid_lambda = lambda_var * stall_factor
                        hybrid_nu = nu
                    
                    cuda.memcpy_htod(lambda_gpu, hybrid_lambda.astype(np.float32))
                    cuda.memcpy_htod(nu_gpu, hybrid_nu.astype(np.int32))
                    
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu,
                                   lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                   block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                elif config == 7:  # Multi-Population PSA - 真正的多族群實現
                    # 為每個族群使用不同的參數策略
                    for pop_idx in range(num_populations):
                        rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                        cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                        
                        # 多樣化參數策略
                        if pop_idx == 0:
                            # 保守族群：標準參數
                            pop_lambda = lambda_var
                            pop_delta = delta
                            pop_nu = nu
                        elif pop_idx == 1:
                            # 探索族群：高擾動
                            pop_lambda = lambda_var * 1.8
                            pop_delta = delta * 1.5
                            pop_nu = np.maximum(nu // 2, 1)
                        else:
                            # 精煉族群：低擾動，高精度
                            pop_lambda = lambda_var * 0.6
                            pop_delta = delta * 0.8
                            pop_nu = nu * 2
                        
                        # 根據族群性能動態調整
                        if cycle > 50 and len(population_fitness) == num_populations:
                            if population_fitness[pop_idx] < np.mean(population_fitness):
                                # 表現差的族群增加擾動
                                pop_lambda *= 1.3
                        
                        cuda.memcpy_htod(lambda_gpu, pop_lambda.astype(np.float32))
                        cuda.memcpy_htod(delta_gpu, pop_delta.astype(np.float32))
                        cuda.memcpy_htod(nu_gpu, pop_nu.astype(np.int32))
                        
                        annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, populations[pop_idx], 
                                       rnd_ini_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                       block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                elif config == 8:  # Smart Adaptive PSA - 結合所有智能特徵
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    
                    # 多層自適應調整
                    # 1. 基於能量趨勢的 lambda 調整
                    if len(energy_window) > 20:
                        short_trend = energy_window[-5:] 
                        long_trend = energy_window[-20:-15]
                        momentum = np.mean(short_trend) - np.mean(long_trend)
                        adaptive_factor = 1.0 + np.tanh(momentum / 100) * 0.3
                        smart_lambda = lambda_var * adaptive_factor * current_alpha
                    else:
                        smart_lambda = lambda_var * current_alpha
                    
                    # 2. 基於收斂速度的 delta 調整
                    convergence_rate = abs(best_energy - energy_window[-1]) if energy_window else 1.0
                    smart_delta = delta * (1.0 + convergence_rate / adaptive_target_drop * 0.1)
                    
                    # 3. 智能 nu 調整（基於停滯時間）
                    stagnation_ratio = (cycle - last_improve_cycle) / max(stall_threshold, 1)
                    if stagnation_ratio > 0.5:
                        smart_nu = np.maximum(nu // 2, 1)  # 停滯時更頻繁更新
                    else:
                        smart_nu = nu
                    
                    # 4. 溫度微調（基於整體性能）
                    if len(energy_window) > 10:
                        performance_ratio = (energy_window[0] - energy_window[-1]) / max(adaptive_target_drop, 1)
                        I0_adjusted = I0 * (1.0 + performance_ratio * 0.05)
                    else:
                        I0_adjusted = I0
                    
                    cuda.memcpy_htod(lambda_gpu, smart_lambda.astype(np.float32))
                    cuda.memcpy_htod(delta_gpu, smart_delta.astype(np.float32))
                    cuda.memcpy_htod(nu_gpu, smart_nu.astype(np.int32))
                    
                    annealing_kernel(np.int32(vertex), I0_adjusted, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu,
                                   lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                   block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1
                # === 新增：config 9 和 10 對應 0702版本 ===
                elif config == 9:  # TApSA 0702版本
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    # 使用動態調整的 mean_range
                    annealing_kernel(np.int32(dynamic_mean_range), np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, 
                                   spin_vector_gpu, rnd_ini_gpu, D_res_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                   block=threads_per_block, grid=blocks_per_grid, shared=shared_mem_size)
                    count_device = count_device + 1
                elif config == 10:  # SpSA 0702版本
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    annealing_kernel(np.float32(stall_prop), np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, 
                                   spin_vector_gpu, rnd_ini_gpu, lambda_gpu, delta_gpu, nu_gpu, np.int32(count_device),
                                   block=threads_per_block, grid=blocks_per_grid)
                    count_device = count_device + 1

                # 3. 能量監控和自適應調整（適用於config 5,6,7,8）
                if config in [6, 7, 8] and i % 10 == 0:  # config 5 現在在GPU端處理，不需要主機端監控
                    # 計算當前能量
                    if config == 7:  # 多族群：使用最佳族群的能量
                        best_energy_pop = float('inf')
                        for pop_idx in range(num_populations):
                            temp_spin = np.empty_like(spin_vector)
                            cuda.memcpy_dtoh(temp_spin, populations[pop_idx])
                            pop_energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)
                            if pop_energy < best_energy_pop:
                                best_energy_pop = pop_energy
                        current_energy = best_energy_pop
                    else:  # 單族群演算法
                        temp_spin = np.empty_like(spin_vector)
                        cuda.memcpy_dtoh(temp_spin, spin_vector_gpu)
                        current_energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)
                    energy_window.append(current_energy)
                    
                    # 更新最佳能量和停滯計數
                    if current_energy < best_energy:
                        best_energy = current_energy
                        last_improve_cycle = cycle
                    
                    # 自適應參數調整（config 5 現在在GPU端處理）
                    if config in [6, 7, 8] and len(energy_window) == window_size:
                        # 其他 config 使用完整的自適應調整
                        delta = energy_window[0] - energy_window[-1]
                        # 調整擾動強度
                        current_alpha += eta_alpha * (adaptive_target_drop - delta)
                        current_alpha = max(0.1, min(0.9, current_alpha))
                        # 調整冷卻速度
                        current_beta += adaptive_eta_beta * (adaptive_target_drop - delta)
                        current_beta = max(beta * 0.5, min(beta * 2.0, current_beta))
                    
                    # === 停滯偵測與 cluster-flip ===
                    # Config 5 現在在GPU端處理，不需要主機端cluster-flip
                    if config == 8 and (cycle - last_improve_cycle) > stall_threshold:
                        print(f"Stagnation detected at cycle {cycle}, performing cluster-flip")
                        temp_spin = cluster_flip(temp_spin, J_matrix, flip_probability=0.3)
                        cuda.memcpy_htod(spin_vector_gpu, temp_spin)
                        last_improve_cycle = cycle
                
                # 4. 動態演算法切換（config 6）
                if config == 6 and cycle % 50 == 0 and cycle > 0:
                    # 評估最近50步的性能，切換到最佳演算法
                    if len(energy_window) >= 10:
                        recent_improvement = energy_window[-10] - energy_window[-1]
                        algorithm_performance[current_algorithm] += recent_improvement
                        
                        # 選擇表現最佳的演算法
                        best_algorithm = max(algorithm_performance, key=algorithm_performance.get)
                        if best_algorithm != current_algorithm:
                            print(f"Switching from {current_algorithm} to {best_algorithm} at cycle {cycle}")
                            current_algorithm = best_algorithm
                        
                        # 重置性能計數
                        algorithm_performance = {k: 0 for k in algorithm_performance}
                
                # 5. 多族群遷移和適應度評估（config 7）
                if config == 7 and cycle % migrate_interval == 0 and cycle > 0:
                    # 評估各族群適應度
                    for pop_idx in range(num_populations):
                        temp_spin = np.empty_like(spin_vector)
                        cuda.memcpy_dtoh(temp_spin, populations[pop_idx])
                        pop_energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)
                        population_fitness[pop_idx] = pop_energy
                    
                    # 找出最佳族群和最差族群
                    best_pop_idx = np.argmin(population_fitness)
                    worst_pop_idx = np.argmax(population_fitness)
                    
                    # 精英遷移：最佳族群向最差族群遷移部分個體
                    migrate_elements = max(1, int(vertex * migrate_size))
                    print(f"Cycle {cycle}: Elite migration from population {best_pop_idx} to {worst_pop_idx}")
                    print(f"Population fitness: {[f'{f:.1f}' for f in population_fitness]}")
                    
                    # 實現真正的個體遷移（隨機選擇部分bit進行遷移）
                    best_spin = np.empty_like(spin_vector)
                    worst_spin = np.empty_like(spin_vector)
                    cuda.memcpy_dtoh(best_spin, populations[best_pop_idx])
                    cuda.memcpy_dtoh(worst_spin, populations[worst_pop_idx])
                    
                    # 隨機選擇要遷移的位置
                    migrate_indices = np.random.choice(vertex, migrate_elements, replace=False)
                    worst_spin[migrate_indices] = best_spin[migrate_indices]
                    
                    # 更新最差族群
                    cuda.memcpy_htod(populations[worst_pop_idx], worst_spin)
                
                # 6. 每個 tau 步都記錄 energy（只對最後一個 trial）
                if k == trial:
                    temp_spin = np.empty_like(spin_vector)
                    cuda.memcpy_dtoh(temp_spin, spin_vector_gpu)
                    energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)
                    energy_per_cycle_last_trial.append(energy)

            # === 能量監控（每 tau 步） ===
            temp_spin = np.empty_like(spin_vector)
            cuda.memcpy_dtoh(temp_spin, spin_vector_gpu)
            current_energy = energy_calculate(vertex, h_vector, J_matrix, temp_spin)

            # 更新 window & 最佳能量
            energy_window.append(current_energy)
            if current_energy < best_energy:
                best_energy        = current_energy
                last_improve_cycle = cycle

            # === Adaptive β 調度 ===
            if len(energy_window) == adaptive_window_size:
                ΔE = energy_window[0] - energy_window[-1]
                # 若下降不足，放慢冷卻 (減小 beta)；若下降過快，加快冷卻
                current_beta += adaptive_eta_beta * (adaptive_target_drop - ΔE)
                current_beta = max(beta * 0.5, min(beta * 2.0, current_beta))

            if config not in [8]:  # 傳統溫度調度（config 5 也使用標準調度）
                I0 /= beta
            elif config in [5]:
                I0 /= current_beta
            else:  # 只有 config 8 使用自適應 beta
                I0 /= current_beta
            cycle += 1

        time_end_gpu.record()
        time_end_gpu.synchronize()
        annealing_time = time_start_gpu.time_till(time_end_gpu)

        # === 獲取GPU自適應參數（僅用於Config 5） ===
        if config == 5:
            # 從GPU獲取最終的自適應參數
            alpha_result = np.zeros(1, dtype=np.float32)
            beta_result = np.zeros(1, dtype=np.float32)
            best_energy_result = np.zeros(1, dtype=np.float32)
            
            alpha_gpu = cuda.mem_alloc(alpha_result.nbytes)
            beta_gpu = cuda.mem_alloc(beta_result.nbytes)
            best_energy_gpu_result = cuda.mem_alloc(best_energy_result.nbytes)
            
            get_adaptive_params_kernel(adaptive_state_gpu, alpha_gpu, beta_gpu, best_energy_gpu_result,
                                     block=(1, 1, 1), grid=(1, 1, 1))
            
            cuda.memcpy_dtoh(alpha_result, alpha_gpu)
            cuda.memcpy_dtoh(beta_result, beta_gpu)
            cuda.memcpy_dtoh(best_energy_result, best_energy_gpu_result)
            
            # 更新主機端變數以用於結果輸出
            current_alpha = float(alpha_result[0])
            current_beta = float(beta_result[0])
            best_energy = float(best_energy_result[0])

        # 最終結果計算
        if config == 7:  # 多族群：選擇最佳族群
            best_cut = 0
            best_pop_idx = 0
            for pop_idx in range(num_populations):
                cut_calculate_kernel(np.int32(vertex), J_matrix_gpu, populations[pop_idx], cut_val_gpu, 
                                   block=threads_per_block, grid=blocks_per_grid)
                cuda.memcpy_dtoh(cut_val, cut_val_gpu)
                if cut_val[0] > best_cut:
                    best_cut = cut_val[0]
                    best_pop_idx = pop_idx
            # 使用最佳族群作為最終結果
            spin_vector_gpu = populations[best_pop_idx]

        last_spin_vector = np.empty_like(spin_vector)
        cuda.memcpy_dtoh(last_spin_vector, spin_vector_gpu)
        
        cut_calculate_kernel(np.int32(vertex), J_matrix_gpu, spin_vector_gpu, cut_val_gpu, block=threads_per_block, grid=blocks_per_grid)
        cuda.memcpy_dtoh(cut_val, cut_val_gpu)

        min_energy = energy_calculate(vertex, h_vector, J_matrix, last_spin_vector)

        cut_list.append(int(cut_val[0]))
        time_list.append(annealing_time)
        energy_list.append(min_energy)

        print('Graph:', file_base)
        print('Time:', annealing_time)
        print('Cut value:', cut_val)
        print('Ising Energy:', min_energy)
        if config == 5:
            print(f'[GPU Adaptive] Final alpha: {current_alpha:.3f}, Final beta: {current_beta:.3f}')
            print(f'[GPU Adaptive] Best energy: {best_energy:.1f}')
        elif config in [6, 8]:
            print(f'Final alpha: {current_alpha:.3f}, Final beta: {current_beta:.3f}')
            print(f'Best energy: {best_energy:.1f}, Total improvements: {cycle - last_improve_cycle}')
        elif config == 7:
            print(f'Population fitness: {[f"{f:.1f}" for f in population_fitness]}')

        # === Config 5 記憶體清理 ===
        if config == 5:
            # 清理臨時分配的記憶體
            alpha_gpu.free()
            beta_gpu.free()
            best_energy_gpu_result.free()
            # adaptive_state_gpu 和 rand_states_gpu 在每個trial結束時會自動清理

    # trial 迴圈結束後，畫出能量圖並儲存為 Excel
    plt.figure(figsize=(10,6))
    plt.plot(energy_per_cycle_last_trial)
    plt.xlabel('Cycles')
    plt.ylabel('Energy')
    plt.title('Energy vs Cycles (Last Trial)')
    plt.grid(True)
    plt.tight_layout()
    # 建立資料夾並加上時間戳記和參數到檔名
    os.makedirs('./0708_energy_plots', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 儲存為 PNG
    png_filename = f'./0708_energy_plots/energy_vs_cycles_{file_base}_config{config}_tau{tau}_res{res}_l{l_scale}_d{d_scale}_n{n_scale}_{timestamp}.png'
    plt.savefig(png_filename)
    
    # 儲存為 Excel
    excel_filename = f'./0708_energy_plots/energy_vs_cycles_{file_base}_config{config}_tau{tau}_res{res}_l{l_scale}_d{d_scale}_n{n_scale}_{timestamp}.xlsx'
    
    # 建立 DataFrame
    df = pd.DataFrame({
        'Cycle': range(len(energy_per_cycle_last_trial)),
        'Energy': energy_per_cycle_last_trial
    })
    
    # 使用 ExcelWriter 建立 Excel 檔案並加入圖表
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 寫入數據到 Excel
        df.to_excel(writer, sheet_name='Energy_Data', index=False)
        
        # 取得 worksheet 並加入圖表
        workbook = writer.book
        worksheet = writer.sheets['Energy_Data']
        
        # 建立圖表
        from openpyxl.chart import LineChart, Reference
        chart = LineChart()
        chart.title = "Energy vs Cycles (Last Trial)"
        chart.style = 13
        chart.x_axis.title = 'Cycles'
        chart.y_axis.title = 'Energy'
        
        # 設定數據範圍
        data = Reference(worksheet, min_col=2, min_row=1, max_row=len(energy_per_cycle_last_trial)+1, max_col=2)
        cats = Reference(worksheet, min_col=1, min_row=2, max_row=len(energy_per_cycle_last_trial)+1)
        
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        
        # 將圖表加入到 worksheet
        worksheet.add_chart(chart, "D2")
    
    plt.close()
    print(f"Energy plot saved as PNG: {png_filename}")
    print(f"Energy data and chart saved as Excel: {excel_filename}")

    return cut_list, time_list, energy_list

def save_results(best_known, trial, cut_list, time_list, energy_list, args, file_base, first_line, second_line, third_line, fourth_line,  sigma_vector, I0_min, I0_max, total_time):
    #-----------Statistical processing-----------
    del cut_list[0]
    del time_list[0]
    del energy_list[0]

    cut_average = sum(cut_list) / trial
    cut_max = max(cut_list)
    cut_min = min(cut_list)
    std_cut = statistics.stdev(cut_list)
    time_average = sum(time_list) / trial
    energy_average = sum(energy_list) / trial
    P = sum(1 for x in cut_list if x > best_known * 0.99) / trial
    if P == 1:
        TTS = time_average
    elif P == 0:
        TTS = 'None'
    else:
        TTS = time_average * math.log(1 - 0.99) / math.log(1 - P)

    print('######################## Final result #######################')
    print('Average cut:', cut_average)
    print('Maximum cut:', cut_max)
    print('Minimum cut:', cut_min)
    print('Average annealing time:', time_average, "[ms]")
    print('Average energy:', energy_average)
    print('TTS(0.99):', TTS)
    print('Average reachability [%]:', 100 * cut_average / best_known)
    print('Maximum reachability [%]:', 100 * cut_max / best_known)
    print('Std of cut value:', std_cut)

    if args.config == 1:
        alg = 'SSA'
    elif args.config == 2:
        alg = 'pSA'
    elif args.config == 3:
        alg = 'TApSA'
    elif args.config == 4:
        alg = 'SpSA'
    elif args.config == 5:
        alg = 'ApSA'    # Adaptive PSA - 自適應溫度調度
    elif args.config == 6:
        alg = 'HpSA'    # Hybrid PSA - 混合演算法框架
    elif args.config == 7:
        alg = 'MpSA'    # Multi-Population PSA - 多族群演算法
    elif args.config == 8:
        alg = 'SmartPSA'  # Smart Adaptive PSA - 智能自適應演算法
    elif args.config == 9:
        alg = 'TApSA0702'  # TApSA 0702版本 - 對應 gpu_MAXCUT_var0702.py config 3
    elif args.config == 10:
        alg = 'SpSA0702'   # SpSA 0702版本 - 對應 gpu_MAXCUT_var0702.py config 4

    csv_file_name1 = f'./result/{alg}_result_unique{args.unique}_config{args.config}_cycle{args.cycle}_trial{args.trial}_tau{args.tau}_thread{args.thread}_param{args.param}_res{args.res}.csv'
    csv_file_name2 = f'./result/{alg}_cut_unique{args.unique}_config{args.config}_cycle{args.cycle}_trial{args.trial}_tau{args.tau}_thread{args.thread}_param{args.param}_res{args.res}.csv'

    data = [
        file_base, first_line, second_line, third_line, fourth_line, cut_average, cut_max, cut_min, std_cut, 0.67448975 * np.mean(sigma_vector), I0_min, I0_max, 100 * cut_average / int(fourth_line), 100 * cut_max / int(fourth_line), time_average, total_time, args.mean_range, args.stall_prop, args.l_scale, args.d_scale, args.n_scale]

    if os.path.isfile(csv_file_name1):
        with open(csv_file_name1, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
    else:
        with open(csv_file_name1, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Gset', 'number of edges', 'edge value', 'edge type', 'best-known value', 'mean_cut_value', 'max_cut_value', 'min_cut_value', 'std_cut', 'n_rnd', 'I0_min', 'I0_max', 'ratio of mean/best', 'ratio of max/best', '1 annealing_time [ms]', 'Total time [s]', 'mean_range', 'stall_prop', 'l_scale', 'd_scale', 'n_scale'])
            writer.writerow(data)

    with open(csv_file_name2, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([cut_average, cut_max, cut_min, std_cut, time_average])

def main():
    starttime = time.time()
    rs = time.time()
    #verbose = True

    args = parse_arguments()
    device = initialize_cuda(args)
    config = args.config
    gpu_code = load_gpu_code(config)
    
    first_line, second_line, third_line, fourth_line, lines = read_file_MAXCUT(args.file_path)
    vertex = int(first_line)
    G_matrix = get_graph_MAXCUT(vertex, lines)
    J_matrix = (-G_matrix).astype(np.int32)
    h_vector = np.reshape(np.diag(J_matrix), (vertex, 1))
    file_path = args.file_path
    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    re = time.time()
    print('File reading time:', re - rs, '[s]')
    
    best_known = np.int32(fourth_line)

    min_cycle, trial, tau, Mshot, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid = set_annealing_parameters(vertex, args, h_vector, J_matrix)

    print(f"blocks_per_grid: {blocks_per_grid}")
    print(f"threads_per_block: {threads_per_block}")
    print(f"vertex: {vertex}, tau: {tau}, beta: {beta}, I0_min: {I0_min}, I0_max: {I0_max}")

    try:
        cut_list, time_list, energy_list = run_trials(args, file_base, config, vertex, min_cycle, trial, tau, Mshot,  gpu_code, h_vector, G_matrix, J_matrix, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid)
    finally:
        cleanup_cuda()

    total_time = time.time() - starttime
    save_results(best_known, trial, cut_list, time_list, energy_list, args, file_base, first_line, second_line, third_line, fourth_line, sigma_vector, I0_min, I0_max, total_time)
    print("Total time:", total_time, '[s]')

if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())