# -*- codeing = utf-8 -*-
# @Time : 2024/10/9 16:40
# @Author : Luo_CW
# @File : main.py
# @Software : PyCharm
import os
# import gensim
import jieba
import pickle as pkl
def train_word():

    # 获取每一行数据
    sentence_datas = open('data/test.txt','r',encoding='utf-8').read().split('\n')
    # 将句子划分为单个字
    # word_datas = [[i for i in data[:-2] if i!= " "] for data in sentence_datas]
    # 得到一行的单个字 [[],...,[]]
    word_datas = []
    for data in sentence_datas:
        word_data = []
        for i in data[:-2]:
            if i != " ":
                word_data.append(i)
        word_datas.append(word_data)

    model = Word2Vec(
        word_datas, #需要训练的数据
        vector_size = 10, #词向量的维度
        window = 2,  # 居中当前单词与预测单词之间的最大距离
        win_count = 1, # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
        workers = 8,  # 使用这些工作线程来训练模型（使用多核机器进行更快的训练）
        sg=0,
        epochs = 10
    )


if __name__ == '__main__':
    train_word()