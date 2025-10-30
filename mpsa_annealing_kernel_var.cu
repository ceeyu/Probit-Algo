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

    // Multi-Population PSA kernel - 使用與原始PSA相同的參數格式
    // 族群參數差異在主機端生成，保持kernel簡潔性
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
                
                // 標準的變異數公式（族群差異已在主機端完成）
                float Itanh = tanh(lambda[i] * mem_I0 * (D_res + delta[i])) + rnd[i];
                spin_vector[i] = (Itanh > 0) ? 1 : -1;
                
                __syncthreads();
            }
        }
    }

    // 族群遷移kernel - 用於在不同族群間交換優秀個體
    __global__ void population_migration(int vertex, int* source_population, int* target_population, float* fitness_scores, int migrate_size)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < migrate_size && i < vertex)
        {
            // 簡化的遷移策略：直接複製前migrate_size個個體
            target_population[i] = source_population[i];
        }
    }

    // 評估族群適應度kernel
    __global__ void evaluate_population_fitness(int vertex, int8_t* J_matrix, int* spin_vector, float* fitness)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        
        extern __shared__ float shared_energy[];
        shared_energy[threadIdx.y] = 0.0f;
        
        if (i < vertex)
        {
            // 計算局部能量貢獻
            for (int j = 0; j < vertex; j++)
            {
                if (i != j)
                {
                    shared_energy[threadIdx.y] += static_cast<int>(J_matrix[i * vertex + j]) * spin_vector[i] * spin_vector[j];
                }
            }
        }
        
        __syncthreads();
        
        // 歸約計算總適應度
        if (threadIdx.y == 0)
        {
            float total_fitness = 0.0f;
            for (int k = 0; k < blockDim.y; k++)
            {
                total_fitness += shared_energy[k];
            }
            atomicAdd(fitness, -total_fitness / 2.0f); // 負號因為我們要最大化cut
        }
    }
} 