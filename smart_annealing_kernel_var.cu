#include <cuda_runtime.h>

extern "C"
{
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

    // Smart Adaptive PSA kernel - 使用與原始PSA相同的參數格式
    // 智能自適應調整在主機端完成，保持GPU kernel的簡潔性
    __global__ void annealing_module(int vertex, float mem_I0, int8_t *h_vector, int8_t *J_matrix, int *spin_vector, float *rnd, float *lambda, float *delta, int *nu, int count_device)
    {
        int i, k;
        float D_res;

        i = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < vertex)  // ✅ 所有有效線程都進入
        {
            if (count_device % nu[i] == 0)  // ✅ 使用變異數參數nu控制更新頻率
            {
                D_res = h_vector[i];
                __syncthreads();
                for(k=0; k<vertex; k++){
                    D_res += static_cast<int>(J_matrix[i * vertex + k]) * spin_vector[k];
                }
                
                // 標準的變異數公式（智能自適應調整已在主機端完成）
                float Itanh = tanh(lambda[i] * mem_I0 * (D_res + delta[i])) + rnd[i];
                spin_vector[i] = (Itanh > 0) ? 1 : -1;
                
                __syncthreads();
            }
        }
    }

    // 智能停滯檢測kernel - 檢測局部能量是否停滯
    __global__ void detect_stagnation(int vertex, int8_t* J_matrix, int* spin_vector, float* stagnation_metric)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        extern __shared__ float shared_metric[];
        shared_metric[threadIdx.y] = 0.0f;
        
        if (i < vertex)
        {
            // 計算局部能量梯度作為停滯指標
            float local_gradient = 0.0f;
            for (int j = 0; j < vertex; j++)
            {
                if (i != j)
                {
                    local_gradient += fabs(static_cast<float>(J_matrix[i * vertex + j])) * fabs(spin_vector[i] - spin_vector[j]);
                }
            }
            shared_metric[threadIdx.y] = local_gradient / vertex;
        }
        
        __syncthreads();
        
        // 歸約計算總停滯度量
        if (threadIdx.y == 0)
        {
            float total_metric = 0.0f;
            for (int k = 0; k < blockDim.y; k++)
            {
                total_metric += shared_metric[k];
            }
            atomicAdd(stagnation_metric, total_metric);
        }
    }

    // 智能cluster-flip kernel - 基於能量分析的智能翻轉（簡化版）
    __global__ void smart_cluster_flip(int vertex, int8_t* J_matrix, int* spin_vector, float flip_intensity)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < vertex)
        {
            // 計算翻轉該節點的能量變化
            float energy_change = 0.0f;
            for (int j = 0; j < vertex; j++)
            {
                if (i != j)
                {
                    energy_change += static_cast<int>(J_matrix[i * vertex + j]) * spin_vector[j];
                }
            }
            
            // 如果翻轉能降低能量，則進行翻轉
            // 使用簡化的決策機制避免複雜的隨機數生成
            if (energy_change * spin_vector[i] > 0 && (i % 10) < (flip_intensity * 10)) {
                spin_vector[i] *= -1;
            }
        }
    }
} 