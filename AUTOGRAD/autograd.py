import torch


# 입력이 x, 매개변수가 w와 b(weight, bias), loss function이 있ㄷ는 간단한 단일 계층 신경망
# wx+b=z, loss = z(예측값), y(실제 값)의 cross entropy를 통한 loss 값
# requires_grad는 loss function의 변화도를 계산하기 위해 필요하다.
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


# Gradient 계산하기
## optimization을 위해서는 loss function의 derivative를 계산해야 한다. 이를 위해 loss.backward()를 호출하고, w.grad와 b.grad에서 값을 가져온다.
loss.backward()
print(w.grad)
print(b.grad)


# Gradient 추적 멈추기
## requires_grad가 활성화된 Tensor들은 연산 기록을 추적하고 Gradient 계산을 지원한다. 하지만, 모델 학습 이후에 forward prop만 하는 경우, 이 추적이 필요하지 않게 된다. 이 때, torch.no_grad()를 통해 추적을 멈출 수 있게 된다.
z = torch.matmul(x, w)+b
print(z.requires_grad)          # 결과값: True(requires_grad가 True임)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)          # 결과값: False(requires_grad가 False임)

# detach()를 통해서도 동일한 결과를 얻을 수 있다.
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)      # 결과값: False(requires_grad가 False임)
