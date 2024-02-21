import torch
import numpy as np


# Tensor 초기화
# 1. data로부터 직접 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. Numpy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. 다른 tensor로부터 생성
x_ones = torch.ones_like(x_data)	# x 속성 유지
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)	# x 속성 무시(덮어쓰기)
print(f"Random Tensor: \n {x_rand} \n")

# 4. random 혹은 constant 값 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeors Teonsor: \n {zeros_tensor} \n")
