import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Class define
## 신경망 모델을 nn.Module의 하위 클래스로 정의하고, __init__ 에서 신경망 계층 초기화. 상속받은 클래스들은 forward method에 입력 데이터에 대한 연산을 구현해야 한다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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

model = NeuralNetwork().to(device)
print(model)

# model에 입력을 전달하여 호출하면 2차원 tensor을 반환한다. 원시 예측값은 softmax를 통해 예측 확률을 얻는다.
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# Model Layer
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten
## Tensor를 1차원으로 평탄화하여 다층 퍼셉트론 등의 신경망 레이어에 입력으로 제공할 수 있도록 함.
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear
## weight와 bias를 이용하여 선형 변환을 적용하는 모듈
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
## activation function. 비선형성을 도입하여 신경망이 여러 현상에 대해 학습할 수 있도록 함.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
## 순서를 갖는 모듈의 컨테이너. Data는 정의된 것과 같은 순서로 모든 모듈을 통해 전달.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax
## logits는 [-inf, inf] 범위로 전달되는데, 이를 모델의 각 class에 대한 예측 확률을 나타낼 수 있도록 [0, 1] 범위로 scaling하는 함수. dim 매개변수는 값의 합이 1이 되는 차원을 나타냄.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# model parameter
## 신경망 내부의 많은 layer들은 parameterize된다. 최적화되는 weight, bias와 연관이 되며, nn.Module을 상속하면 내부의 모든 필드들이 자동으로 track된다.
## 해당 예제는 각 parameter를 iterate하여 크기와 값을 출력한다.
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
