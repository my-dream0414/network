# -*- codeing = utf-8 -*-
# @Time : 2024/10/9 17:12
# @Author : Luo_CW
# @File : main.py
# @Software : PyCharm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import read_data, built_curpus, TextDataset
from model import TextCNNModel
from config import parsers
import pickle as pkl
from sklearn.metrics import accuracy_score
import time
from test import test_data

if __name__ == '__main__':
    start = time.time()
    args = parsers()
    # 获取数据
    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.data_pkl):
        dataset = pkl.load(open(args.data_pkl, "rb"))
        word_2_index, words_embedding = dataset[0], dataset[1]
    else:
        word_2_index, words_embedding = built_curpus(train_text, args.embedding_num)

    # 训练部分
    train_dataset = TextDataset(train_text, train_label, word_2_index, args.max_len)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    # 测试部分
    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, args.max_len)
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False)

    # 计算损失
    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    # 实现 Adam 优化算法的类
    opt = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    acc_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0, 0
        for batch_index, (batch_text, batch_label) in enumerate(train_loader):
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)

            loss = loss_fn(pred, batch_label)
            # 将优化器管理的所有参数的梯度归零
            opt.zero_grad()
            # 用于计算‌损失函数loss关于模型参数的梯度
            # 会自动计算损失函数相对于模型参数的梯度，并将这些梯度存储在模型的参数中
            loss.backward()
            # 是PyTorch优化器对象的一个方法，用于根据梯度和学习率更新模型参数，以最小化损失函数。
            opt.step()

            loss_sum += loss
            count += 1

            # 打印内容
            if len(train_loader) - batch_index <= len(train_loader) % 1000 and count == len(train_loader) % 1000:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0
        # 在评估模式下，使用model.eval()会关闭Batch Normalization和Dropout
        model.eval()
        all_pred, all_true = [], []
        # 在 torch.no_grad() 上下文中禁用梯度计算
        # 将不需要梯度计算的代码块放在 with torch.no_grad(): 下即可。
        with torch.no_grad():
            for batch_text, batch_label in dev_loader:

                # 数据迁移语句：
                # 在执行数据迁移操作。这个表达式会立即执行，
                # 将data_pet的数据从当前设备（可能是CPU）移动到指定的device（可能是GPU或其他设备）。
                # 这通常用于将模型或数据加载到GPU上进行加速计算。
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                pred = model(batch_text)

                # 用于返回指定维度上最大值的索引
                # dim参数指定了要减少的维度，keepdim参数决定了输出张量的维度是否与输入张量相同。
                pred = torch.argmax(pred, dim=1)
                # 将PyTorch张量（Tensor）转换为NumPy数组，然后再将NumPy数组转换为Python列表。
                pred = pred.cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        acc = accuracy_score(all_pred, all_true)
        print(f"dev acc:{acc:.4f}")

        if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), args.save_model_best)
            print(f"以保存最佳模型")
        print("*" * 50)

    # 用于保存模型的状态、张量或优化器的状态等。
    # 通过这个函数，我们可以将训练过程中的关键信息持久化，以便在后续的时间里重新加载并继续使用。
    # 将PyTorch对象（如模型、张量等）保存到磁盘上，以文件的形式进行存储。
    torch.save(model.state_dict(), args.save_model_last)

    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f} min")
    test_data()