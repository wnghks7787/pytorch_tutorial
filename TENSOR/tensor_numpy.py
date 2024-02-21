import torch
import numpy as np


# tensor -> numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# tensor 변경 사항이 numpy에 반영된다.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


# numpy -> tensor
n = np.ones(5)
t = torch.from_numpy(n)

# numpy 변경 사항이 tensor에 반영된다.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
