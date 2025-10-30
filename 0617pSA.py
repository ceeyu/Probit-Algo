import numpy as np

# 假設初始參數
vertex = 4
mem_I0 = 1.0
h_vector = np.array([0.0, 0.0, 0.0, 0.0])
J_matrix = np.array([
    [0, 1, -1, 0],
    [1, 0, 1, -1],
    [-1, 1, 0, 1],
    [0, -1, 1, 0]
])
spin_vector = np.array([1, -1, 1, -1])
rnd = np.random.normal(0, 0.1, vertex)
lambda_ = np.ones(vertex)
delta = np.zeros(vertex)
nu = np.ones(vertex, dtype=int)
count_device = 2

print('初始 spin_vector:', spin_vector)
print('\nJ_matrix 全部內容:')
print(J_matrix)

for i in range(vertex):
    print(f'\n--- 處理第 {i} 個 p-bit ---')
    print(f'  J_matrix 第 {i} 行: {J_matrix[i, :]}')
    mod_val = count_device % nu[i]
    print(f'  count_device % nu[{i}] = {count_device} % {nu[i]} = {mod_val}')
    if mod_val == 0:
        D_res = h_vector[i]
        print(f'  初始化 D_res = {D_res}')
        for k in range(vertex):
            print(f'    J_matrix[{i}][{k}] = {J_matrix[i][k]}, spin_vector[{k}] = {spin_vector[k]}')
            D_res += J_matrix[i][k] * spin_vector[k]
            print(f'    D_res 累加 J_matrix[{i}][{k}] * spin_vector[{k}] = {J_matrix[i][k]} * {spin_vector[k]} -> D_res = {D_res}')
        print(f'  處理完第 {i} 行後 D_res = {D_res}')
        print(f'  J_matrix 第 {i} 列: {J_matrix[:, i]}')
        Itanh = np.tanh(lambda_[i] * mem_I0 * (D_res + delta[i])) + rnd[i]
        print(f'  Itanh = tanh({lambda_[i]} * {mem_I0} * ({D_res} + {delta[i]})) + {rnd[i]} = {Itanh}')
        if Itanh > 0:
            spin_vector[i] = 1
        else:
            spin_vector[i] = -1
        print(f'  更新 spin_vector[{i}] = {spin_vector[i]}')
    else:
        print(f'  跳過第 {i} 個 p-bit (因為 count_device % nu[{i}] != 0)')

print('\n最終 spin_vector:', spin_vector)