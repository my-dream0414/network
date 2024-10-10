# -*- codeing = utf-8 -*-
# @Time : 2024/9/13 16:46
# @Author : Luo_CW
# @File : InputCoder.py
# @Software : PyCharm
import math

import torch
import torch.nn as nn
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Embeddings(nn.Module):
    """
    类的初始化：
    :param  d_model:词向量的维度，512
    :param  vocab:当前语言的词表大小
    """
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        # 调用nn.Embedding 预定义层，获得实例化词嵌入对象self.lut
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    def forward(self,x):
        """
        Embedding层的前向传播
        :param x:输入给模型的单词文本通过此表映射后的one-hot向量
        x传给self.lut，得到形状为(batch_size, sequence_length, d_model)的张量，与self.d_model相乘，
        以保持不同维度间的方差一致性，及在训练过程中稳定梯度
        :return:
        """
        return self.lut(x) * math.sqrt(self.d_model)
model = Embeddings().to(device)

word = "你是一个老师！"
output = model(word)
print(output)

