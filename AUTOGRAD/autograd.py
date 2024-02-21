import torch


# 입력이 x, 매개변수가 w와 b(weight, bias), loss function이 있ㄷ는 간단한 단일 계층 신경망
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

