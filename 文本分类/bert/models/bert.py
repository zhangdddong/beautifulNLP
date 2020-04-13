#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = 'bert'

        self.train_path = dataset + '/data/train.txt'
        self.test_path = dataset + '/data/test.txt'
        self.dev_path = dataset + '/data/dev.txt'

        # 类别
        with open(dataset + '/data/class.txt', encoding='UTF-8') as f:
            self.class_list = [x.strip() for x in f]

        # 模型保存路径，模型训练结果。
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000batch效果还没有提升，提前结束训练
        self.require_improvement = 1000

        self.num_classes = len(self.class_list)     # 类别数量
        self.num_epochs = 3     # epoch数
        self.batch_size = 128   # batch_size
        self.pad_size = 32  # 每句话处理的长度（短填，长切）
        self.learning_rate = 1e-5   # 学习率

        self.bert_path = 'bert_pretrain'    # bert预训练模型位置
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # bert切词器
        self.hidden_size = 768  # bert隐藏层的个数


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True     # True表示参数不固定，微调  False表示参数固定

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        """
        :param x: input_ids, token_type_ids, attention_mask
        :return: [batch_size, num_classes]
        """
        context = x[0]  # 对应输入的句子 [batch_size -> 128, sentence_length -> 32]
        mask = x[2]     # 负责挖空，对padding部分进行mask [batch_size -> 128, sentence_length -> 32]
        # pooled: [batch_size -> 128, hidden_dim -> 768]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)   # [batch_size -> 128, num_classes -> 10]
        return out
