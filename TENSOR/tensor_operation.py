import torch
import numpy as np

tensor = torch.ones(3, 3)

# GPU가 존재하면, Tensor 이용
if torch.cuda.is_available():
	tensor = tensor.to("cuda")


# Numpy 식의 표준 indexing, slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)


# tensor 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


# arithmetic operation
## 두 tensor 간 행렬 곱 계산(y1, y2, y3는 모두 같은 값이다.)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

## element-wise product 계산(z1, z2, z3은 모두 같은 값이다.)
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(z1)
torch.mul(tensor, tensor, out=z3)


# single-element tensor
## tensor의 모든 값을 하나로 합쳐서 추출
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# in-place 연산
## 연산 결과를 피연산자에 저장. 접미사로 '_'를 가짐.(x.copy_(y) 등)
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
