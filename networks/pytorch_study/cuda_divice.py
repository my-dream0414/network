# -*- codeing = utf-8 -*-
# @Time : 2024/9/9 20:53
# @Author : Luo_CW
# @File : cuda_divice.py
# @Software : PyCharm
import torch
# 测试GPU环境是否可用
# 输出版本
print(torch.__version__)
# 输出cuda版本
print(torch.version.cuda)
# 查看cuda是否可以
print(torch.cuda.is_available())
# 1.11.0
# 11.3
# True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import torch.nn as nn


# Softmax 函数是一种常用的激活函数，通常用于多分类问题中。它将一个含有多个实数值的向量（通常称为 logits）转换成一个概率分布，使得每个元素都在 (0, 1) 区间内，并且所有元素的和为 1。
# 其中 dim=0 对应于每个类的 10 个原始预测值的每个输出，dim=1 对应于每个输出的各个值。
# 创建一个 3x4 的输入张量
input_tensor = torch.randn(3, 4)

# 创建 Softmax 层（dim（可选）：指定 softmax 函数计算的维度。默认值为 -1，表示最后一个维度。）
softmax_layer = nn.Softmax(dim=1)

# 对输入张量应用 Softmax 层
output_tensor = softmax_layer(input_tensor)

print(output_tensor)