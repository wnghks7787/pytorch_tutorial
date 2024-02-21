import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# ToTensor(): PIL image 및 Numpy를 FloatTensor로 변환하고, image pixel 크기를 [0., 1.] 범위로 scaling
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Lambda Transform
# lambda(사용자 정의 함수)를 적용. 아래 함수는 정수를 one-hot tensor로 바꾸는 작업을 진행함.
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
