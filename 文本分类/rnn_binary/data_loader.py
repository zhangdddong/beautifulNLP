#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import numpy as np

from data_utils import clean_str


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    加载数据集和标签
    :param positive_data_file: 正例文件
    :param negative_data_file: 负例文件
    :return: x_text: [batch_size] y: [batch_size, 2]
    """
    # 加载数据
    with open(positive_data_file, encoding='UTF-8') as f:
        positive_examples = f.readlines()
        positive_examples = [s.strip() for s in positive_examples]
    with open(negative_data_file, encoding='UTF-8') as f:
        negative_examples = f.readlines()
        negative_examples = [s.strip() for s in negative_examples]

    # 合并数据集
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # 生成标签
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], axis=0)

    return [x_text, y]


if __name__ == '__main__':
    pos_dir = './data/rt-polarity.pos'
    neg_dir = './data/rt-polarity.neg'

    load_data_and_labels(pos_dir, neg_dir)
