# -*- codeing = utf-8 -*-
# @Time : 2024/10/9 19:03
# @Author : Luo_CW
# @File : test.py
# @Software : PyCharm
import torch
from utils import read_data, TextDataset
from config import parsers
from torch.utils.data import DataLoader
from model import TextCNNModel
from sklearn.metrics import accuracy_score
import pickle as pkl

def test_data():
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = pkl.load(open(args.data_pkl,'rb'))
    word_2_index, words_embedding = dataset[0], dataset[1]
    # 读取带标签数据
    test_text, test_label = read_data(args.test_file)
    test_dataset = TextDataset(test_text, test_label, word_2_index, args.max_len)
    # DataLoader的参数：test_dataset：要加载的数据集|batch_size：指定每个批次的大小（表示从数据集中多少个数据）
    # shuffle：为True表示每个epoch开始时打乱数据，一个epoch意味着数据加载器将整个数据集按照指定的顺序遍历一次
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in test_dataloader:
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()

            all_pred.extend(pred)
            all_true.extend(label)

    accuracy = accuracy_score(all_true, all_pred)

    print(f"test dataset accuracy:{accuracy:.4f}")
