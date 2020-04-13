#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train


parser = argparse.ArgumentParser(description='My-Bert-Text-Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'    # 数据集地址

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    # 保证每次运行结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    # torch.backends.cudann.deterministic = True

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)

    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)
    # for i, (trains, labels) in enumerate(train_iter):
    #     print(i, labels)

    time_dif = utils.get_time_dif(start_time)
    print('模型开始之前，准备数据时间', time_dif)

    # 模型训练，评估测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
