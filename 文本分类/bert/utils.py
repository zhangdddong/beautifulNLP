#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

from tqdm import tqdm
import torch
import time
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path, config):
    """
    返回结果4个list: ids, label, ids_len, mask
    :param file_path:
    :param config:
    :return: ids -> list, label -> int, ids_len -> int, mask -> list
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, lable = line.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)

            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + [0] * (pad_size - len(token))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(lable), seq_len, mask))

    return contents


def build_dataset(config):
    """
    返回训练集、测试集、验证集
    每个集合包含4个list: ids, label, id_len, mask
    :param config:
    :return: train dev test
    """
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)

    return train, dev, test


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.residue = False    # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """
        需要得到3个值 [ids, seq_len, mask]
        :param datas: 4个list: ids, label, id_len, mask
        :return: 3个值 [ids, seq_len, mask]
        """
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)     # 样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)   # 标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)     # 每一个分词序列长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    :param train_data:
    :param config:
    :return:
    """
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """
    获取使用的时间
    :param start_time:
    :return: 换算成 秒
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(time_dif))
