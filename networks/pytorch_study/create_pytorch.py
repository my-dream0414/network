# -*- codeing = utf-8 -*-
# @Time : 2024/9/7 9:45
# @Author : Luo_CW
# @File : create_pytorch.py
# @Software : PyCharm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 引入数据
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
# 获取训练设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}device")
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 通过继承nn.Module来定义我们的神经网络，并在__init__中初始化神经网络层。
# 每个nn.Module都在forward方法中实现对输入数据的操作
class createModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # nn.Sequential
        # 是一个有序的模块容器。数据按定义的相同顺序传递给所有模块。
        # 你可以使用顺序容器来组合一个像seq_modules这样的快速网络。
        self.linear_relu_stack = nn.Sequential(
            # 线性层是一个模块，它使用其存储的权重和偏差对输入应用线性变换
            nn.Linear(28*28, 512),
            # 非线性激活函数是在模型的输入和输出之间创建复杂映射的原因。它们应用于线性变换之后以引入_非线性_，帮助神经网络学习各种现象。
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 创建一个createModule的实例，并将其移动到device，然后打印其结构
model = createModule().to(device)

# 不要直接调用 model.forward()！
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 优化模型参数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 在单个训练循环中，模型对训练数据集（分批馈送）进行预测，并将预测误差反向传播以调整模型的参数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# 我们还根据测试数据集检查模型的性能，以确保它正在学习。
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 训练过程经过多次迭代（“时期”）。
# 在每个时期，模型都会学习参数以做出更好的预测。
# 我们在每个时期打印模型的准确性和损失；
# 我们希望看到准确性随着每个时期的增加而增加，而损失减少。
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")