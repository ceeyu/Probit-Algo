#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C"
{
    // GPU端狀態管理結構
    struct AdaptiveState {
        float current_alpha;           // 當前擾動強度
        float current_beta;            // 當前冷卻速度
        float best_energy;            // 目前最佳能量
        int last_improve_cycle;       // 最後改善的週期
        int stall_counter;            // 停滯計數器
        float energy_window[100];     // 能量視窗 (固定大小)
        int window_index;             // 視窗當前索引
        int window_size;              // 視窗大小
        bool window_full;             // 視窗是否已滿
    };

    __global__ void calculate_cut_val(int vertex, int8_t* J_matrix, int* spin_vector, float* cut_val) 
    {
        int idx = blockIdx.y * blockDim.y + threadIdx.y;
        int stride = blockDim.y * gridDim.y;

        // Use shared memory for intermediate sum reduction
        extern __shared__ float shared_cut[]; 
        shared_cut[threadIdx.y] = 0.0f;

        for (int i = idx; i < vertex; i += stride) {
            for (int j = i + 1; j < vertex; j++) {
                shared_cut[threadIdx.y] -= static_cast<int>(J_matrix[i * vertex + j]) * (1.0f - spin_vector[i] * spin_vector[j]);
            }
        }
        __syncthreads();

        // Perform reduction to sum the values in shared memory
        if (threadIdx.y == 0) {
            float block_sum = 0.0f;
            for (int k = 0; k < blockDim.y; k++) {
                block_sum += shared_cut[k]/2;
            }
            atomicAdd(cut_val, block_sum);
        }  
    }

    // GPU端能量計算函數
    __device__ float calculate_energy_gpu(int vertex, int8_t* h_vector, int8_t* J_matrix, int* spin_vector) {
        float h_energy = 0.0f;
        float J_energy = 0.0f;
        
        // 計算 h_energy
        for (int i = 0; i < vertex; i++) {
            h_energy += h_vector[i] * spin_vector[i];
        }
        
        // 計算 J_energy
        for (int i = 0; i < vertex; i++) {
            float local_field = 0.0f;
            for (int j = 0; j < vertex; j++) {
                local_field += static_cast<int>(J_matrix[i * vertex + j]) * spin_vector[j];
            }
            J_energy += (local_field - h_vector[i]) * spin_vector[i];
        }
        
        return -(J_energy / 2.0f + h_energy);
    }

    // GPU端cluster-flip函數
    __device__ void cluster_flip_gpu(int vertex, int8_t* J_matrix, int* spin_vector, 
                                   curandState* rand_state, float flip_probability = 0.3f) {
        int tid = threadIdx.y;
        
        // 找出能量最高的節點
        __shared__ float local_energies[32];  // 假設blockDim.y <= 32
        __shared__ int high_energy_nodes[32];
        
        if (tid < vertex) {
            // 計算局部能量
            float local_energy = 0.0f;
            for (int k = 0; k < vertex; k++) {
                local_energy += static_cast<int>(J_matrix[tid * vertex + k]) * spin_vector[k];
            }
            local_energies[tid] = local_energy * spin_vector[tid];
        } else {
            local_energies[tid] = -1e10f;  // 設定極小值
        }
        
        __syncthreads();
        
        // 選擇前20%高能量節點進行cluster flip
        int flip_count = max(1, vertex / 5);
        if (tid == 0) {
            // 簡化版：隨機選擇節點進行翻轉
            for (int i = 0; i < flip_count; i++) {
                if (curand_uniform(rand_state) < flip_probability) {
                    int idx = curand(rand_state) % vertex;
                    spin_vector[idx] *= -1;
                    
                    // 擴散到鄰居節點
                    for (int j = 0; j < vertex; j++) {
                        if (J_matrix[idx * vertex + j] != 0 && 
                            curand_uniform(rand_state) < flip_probability * 0.5f) {
                            spin_vector[j] *= -1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // 增強版自適應PSA kernel - 優化版本，減少開銷
    __global__ void annealing_module(int vertex, float mem_I0, int8_t *h_vector, int8_t *J_matrix, 
                                   int *spin_vector, float *rnd, float *lambda, float *delta, int *nu, 
                                   int count_device, 
                                   // 新增的自適應參數
                                   AdaptiveState* adaptive_state, 
                                   float target_drop, float eta_alpha, float eta_beta,
                                   int stall_threshold, int cycle, curandState* rand_states)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < vertex) {
            // === 步驟1: 標準PSA更新 ===
            if (count_device % nu[i] == 0) {
                float D_res = h_vector[i];
                
                // 計算局部場
                for(int k = 0; k < vertex; k++){
                    D_res += static_cast<int>(J_matrix[i * vertex + k]) * spin_vector[k];
                }
                
                // 使用當前的自適應lambda (減少記憶體存取)
                float adaptive_lambda = lambda[i] * adaptive_state->current_alpha;
                float Itanh = tanh(adaptive_lambda * mem_I0 * (D_res + delta[i])) + rnd[i];
                spin_vector[i] = (Itanh > 0) ? 1 : -1;
            }
        }
        
        // 減少同步頻率
        if (threadIdx.y == 0) __syncthreads();
        
        // === 步驟2: 能量監控和自適應調整（降低頻率，減少開銷） ===
        if (threadIdx.y == 0 && blockIdx.y == 0 && (count_device % 50 == 0)) {  // 從10改為50，減少5倍計算
            // 簡化的能量計算 - 只計算能量變化趨勢而不是精確值
            float energy_estimate = 0.0f;
            
            // 快速能量估算：只採樣部分節點
            int sample_size = min(vertex, 100);  // 最多採樣100個節點
            for (int idx = 0; idx < sample_size; idx += 10) {  // 每10個採樣一個
                float local_field = h_vector[idx];
                for (int k = 0; k < vertex; k += 5) {  // 稀疏採樣鄰居
                    local_field += static_cast<int>(J_matrix[idx * vertex + k]) * spin_vector[k];
                }
                energy_estimate -= local_field * spin_vector[idx];
            }
            energy_estimate *= (float)vertex / sample_size;  // 按比例縮放
            
            // 更新能量視窗
            adaptive_state->energy_window[adaptive_state->window_index] = energy_estimate;
            adaptive_state->window_index = (adaptive_state->window_index + 1) % adaptive_state->window_size;
            if (!adaptive_state->window_full && adaptive_state->window_index == 0) {
                adaptive_state->window_full = true;
            }
            
            // 更新最佳能量和停滯計數
            if (energy_estimate < adaptive_state->best_energy) {
                adaptive_state->best_energy = energy_estimate;
                adaptive_state->last_improve_cycle = cycle;
                adaptive_state->stall_counter = 0;
            } else {
                adaptive_state->stall_counter++;
            }
            
            // === 簡化的自適應參數調整 ===
            if (adaptive_state->window_full && (count_device % 100 == 0)) {  // 進一步降低調整頻率
                // 計算能量改善幅度
                int start_idx = adaptive_state->window_index;
                int end_idx = (adaptive_state->window_index - 1 + adaptive_state->window_size) % adaptive_state->window_size;
                float energy_drop = adaptive_state->energy_window[start_idx] - adaptive_state->energy_window[end_idx];
                
                // 簡化的調整邏輯
                if (energy_drop < target_drop * 0.3f) {
                    adaptive_state->current_alpha = fminf(0.95f, adaptive_state->current_alpha * 1.05f);
                } else if (energy_drop > target_drop * 1.5f) {
                    adaptive_state->current_alpha = fmaxf(0.05f, adaptive_state->current_alpha * 0.95f);
                }
                
                // 限制alpha範圍，避免極端值
                adaptive_state->current_alpha = fmaxf(0.1f, fminf(0.9f, adaptive_state->current_alpha));
            }
            
            // === 簡化的停滯檢測（移除cluster-flip以減少開銷） ===
            if (adaptive_state->stall_counter > stall_threshold * 2) {  // 提高閾值
                // 簡單重置：增加擾動而不是cluster-flip
                adaptive_state->current_alpha = fminf(0.9f, adaptive_state->current_alpha * 1.2f);
                adaptive_state->stall_counter = 0;
                // 移除printf以提升效能
            }
        }
    }

    // 初始化自適應狀態
    __global__ void init_adaptive_state(AdaptiveState* adaptive_state, int window_size) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            adaptive_state->current_alpha = 0.8f;
            adaptive_state->current_beta = 1.0f;
            adaptive_state->best_energy = 1e10f;
            adaptive_state->last_improve_cycle = 0;
            adaptive_state->stall_counter = 0;
            adaptive_state->window_index = 0;
            adaptive_state->window_size = window_size;
            adaptive_state->window_full = false;
            
            // 初始化能量視窗
            for (int i = 0; i < 100; i++) {
                adaptive_state->energy_window[i] = 0.0f;
            }
        }
    }

    // 獲取當前自適應參數
    __global__ void get_adaptive_params(AdaptiveState* adaptive_state, float* alpha, float* beta, float* best_energy) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *alpha = adaptive_state->current_alpha;
            *beta = adaptive_state->current_beta;
            *best_energy = adaptive_state->best_energy;
        }
    }

    // 初始化隨機數生成器
    __global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            curand_init(seed, idx, 0, &states[idx]);
        }
    }

    // 簡化版PSA kernel - 只執行基本的PSA更新，不進行自適應監控
    __global__ void simple_annealing_module(int vertex, float mem_I0, int8_t *h_vector, int8_t *J_matrix, 
                                           int *spin_vector, float *rnd, float *lambda, float *delta, int *nu, 
                                           int count_device, AdaptiveState* adaptive_state)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < vertex) {
            if (count_device % nu[i] == 0) {
                float D_res = h_vector[i];
                
                // 計算局部場
                for(int k = 0; k < vertex; k++){
                    D_res += static_cast<int>(J_matrix[i * vertex + k]) * spin_vector[k];
                }
                
                // 使用當前的自適應lambda（從adaptive_state讀取）
                float adaptive_lambda = lambda[i] * adaptive_state->current_alpha;
                float Itanh = tanh(adaptive_lambda * mem_I0 * (D_res + delta[i])) + rnd[i];
                spin_vector[i] = (Itanh > 0) ? 1 : -1;
            }
        }
        
        __syncthreads();
    }
} 