# -*- codeing = utf-8 -*-
# @Time : 2024/9/11 19:37
# @Author : Luo_CW
# @File : Scaled_Dot-Product_Attention.py
# @Software : PyCharm
import torch
# 首先定义input：定义三个1x4的input(input1,input2,input3)
x = [
    [1, 0, 1, 0],  # input 1
    [0, 2, 0, 2],  # input 2
    [1, 1, 1, 1]   # input 3
]
x = torch.tensor(x,dtype=torch.float32)

# 初始化权重

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key,dtype=torch.float32)
w_query = torch.tensor(w_query,dtype=torch.float32)
w_value = torch.tensor(w_value,dtype=torch.float32)
# 每个input和三个权重矩阵分别相乘会得到三个新的矩阵分别是key，query，value
# key = input * w_key;  query = input * w_query;  value = input * w_value;
keys = x @ w_key
querys = x @ w_query
values = x @ w_value
print(keys,querys,values)
# tensor([[0., 1., 1.],
#         [4., 4., 0.],
#         [2., 3., 1.]])
# tensor([[1., 0., 2.],
#         [2., 2., 2.],
#         [2., 1., 3.]])
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])

# 计算attenion scores
# 为了获得input1的注意力分数(attention scores)，我们将input1的query与input1、2、3的key的转置分别作点积，得到3个attention scores。
# 同理，我们也可以得到input2和input3的attention scores。(这里简写，没有实现)
attenion_scores1 = querys @ keys.T
print(attenion_scores1)
# tensor([[ 2.,  4.,  4.],
#         [ 4., 16., 12.],
#         [ 4., 12., 10.]])


# 对attention scores作softmax
# 上一步得到了attention scores矩阵后，我们对attention scores矩阵作softmax计算。softmax的作用为归一化，使得其中各项相加后为1。
# 这样做的好处是凸显矩阵中最大的值并抑制远低于最大值的其他分量。
from torch.nn.functional import softmax

attn_scores_softmax = softmax(attenion_scores1, dim=-1)
print(attn_scores_softmax)
# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)
print(attn_scores_softmax)
# tensor([[0.0000, 0.5000, 0.5000],
#         [0.0000, 1.0000, 0.0000],
#         [0.0000, 0.9000, 0.1000]])

# 将attention scores与values相乘
# 每个score乘以其对应的value得到3个alignment vectors。在本教程中，我们将它们称为weighted values。
weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
print("1:",weighted_values)
# tensor([[[0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000]],
#
#         [[1.0000, 4.0000, 0.0000],
#          [2.0000, 8.0000, 0.0000],
#          [1.8000, 7.2000, 0.0000]],
#
#         [[1.0000, 3.0000, 1.5000],
#          [0.0000, 0.0000, 0.0000],
#          [0.2000, 0.6000, 0.3000]]])
# 对weighted values求和得到output
# 从图中可以看出，每个input生成3个weighed values，我们将这3个weighted values相加，得到output。
# 图中一共有3个input，所以最终生成3个output。
outputs = weighted_values.sum(dim=0)
print(outputs)
# tensor([[2.0000, 7.0000, 1.5000],
#         [2.0000, 8.0000, 0.0000],
#         [2.0000, 7.8000, 0.3000]])