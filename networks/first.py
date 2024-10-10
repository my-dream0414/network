# -*- codeing = utf-8 -*-
# @Time : 2024/9/5 9:54
# @Author : Luo_CW
# @File : first.py
# @Software : PyCharm

import numpy as np
# 定义激活函数sigmoid()
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 初始化模型参数
# 模型参数主要包括两个: w:权重 b:偏置值
def initilize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0.0
    return w,b

# 模型主题部分：前向传播->计算损失->反向传播->权值更新

# 前向传播
def propapate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T)+b)  # np.dot():矩阵乘法函数
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))  # np.log():计算数组中的每个元素的自然对数
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {
        "dw":dw,
        "db":db
    }
    return grads,cost