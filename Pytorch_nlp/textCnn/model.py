# -*- codeing = utf-8 -*-
# @Time : 2024/10/9 17:16
# @Author : Luo_CW
# @File : model.py
# @Software : PyCharm
import torch.nn as nn
import torch
# 创建一个卷积核
class Block(nn.Module):
    def __init__(self,kernel_s,embeddin_num,max_len,hidden_num):
        super().__init__()
        # 创建卷积层
        self.cnn = nn.Conv2d(in_channels=1,out_channels=hidden_num,kernel_size=(kernel_s,embeddin_num))
        # 创建激励层（激活函数）
        self.act = nn.ReLU()
        # 创建池化层(取最大值)
        self.mxp = nn.MaxPool1d(kernel_size=(max_len-kernel_s+1))
    def forward(self,batch_emb): # shape [batch *  in_channel * max_len * emb_num]
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim = -1)
        m = self.mxp(a)
        m = m.squeeze(dim = -1)
        return m

class TextCNNModel(nn.Module):
    def __init__(self,emb_matrix,max_len,class_num,hidden_num):
        super().__init__()
        # 第二个参数为文字维度，一直对应到语料库第二个返回值的第二个元素，承接上文提到的nn.embedding函数的使用与返回结果
        self.emb_num = emb_matrix.weight.shape[1]

        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)
        # self.block4 = Block(5, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num*3,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_idx):
        batch_emb = self.emb_matrix(batch_idx)#根据id号找到相应生成的embedding
        b1_result = self.block1(batch_emb)
        b2_result = self.block2(batch_emb)
        b3_result = self.block3(batch_emb)
        # b4_result = self.block4.forward(batch_emb)
        #拼接
        feature = torch.cat([b1_result,b2_result,b3_result],dim=1)
        pre = self.classifier(feature)

        return pre
