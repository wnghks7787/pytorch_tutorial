# Pre-requisite code

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


# Hyperparameter
## Hyperparameter는 최적화를 제어. 다른 hyperparameter은 모델 학습과 수렴률에 영양을 미침.
## epoch: 데이터셋을 반복하는 횟수
## bathc size: parameter 갱신 전, 신경망을 통해 전파된 데이터 샘플 수
## learning rate: 각 batch/epoch에서 parameter를 조절하는 비율. 값이 너무 작으면 학습 속도가 느려지고, 값이 너무 크면 예측할 수 없는 동작이 발생함.
learning_rate = 1e-3
batch_size = 64
epochs = 5


# Optimization loop
## 최적화의 각 단계를 epoch라고 부름. epoch는 각각 두개의 파트로 구성.
## train loop: training dataset을 iterate(반복)하고, 매개변수를 최적점으로 수렴
## validation/test loop: model 성능 개선 여부를 확인하기 위해 test dataset을 itereate(반복)


# Loss function
## 일반적으로, regression에서는 nn.MSELoss()를 사용한다.
## 일반적으로, classification에서는 nn.NLLLose()를 사용한다.
loss_fn = nn.CrossEntropyLoss()


# Optimizer
## model의 오차를 줄이기 위해 parameters를 조정하는 과정.
## 학습 단계에서의 optimizer 방식
### 1. optimizer.zero_grad() --> parameters 변화도 재설정
### 2. loss.backwards() --> 예측 손실 역전
### 3. optimizer.step() --> parameter 조정.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 전체 코드
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # prediction, loss 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
