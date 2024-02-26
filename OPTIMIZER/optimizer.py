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

test_data = datasets(
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

