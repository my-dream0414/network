# -*- codeing = utf-8 -*-
# @Time : 2024/10/9 21:03
# @Author : Luo_CW
# @File : 测试.py
# @Software : PyCharm
import torch

# 创建一个张量，并设置 requires_grad=True 以便记录梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 在 torch.no_grad() 上下文中禁用梯度计算
with torch.no_grad():
    y = x + 2
    print(y)

# 此时，x 的 requires_grad 属性仍然为 True，但 y 的 requires_grad 属性为 False
print("x 的 requires_grad:", x.requires_grad)
print("y 的 requires_grad:", y.requires_grad)
