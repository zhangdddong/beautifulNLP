#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import time
from importlib import import_module
from utils import build_dataset, get_time_dif, build_net_data
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='TextRNN2', type=str, help='choose a model: TextCNN TextRNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'data'

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model

    x = import_module('models.' + model_name)   # 根据不同配置分别导入模型，对应哪个py文件
    config = x.Config(dataset, embedding)       # 对应Config类的__init__方法

    start_time = time.time()
    print('Loading data ...')
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    time_dif = get_time_dif(start_time)
    print('Time usage: ', time_dif)

    # padding
    train_x, train_y = build_net_data(train_data, config)
    dev_x, dev_y = build_net_data(dev_data, config)
    test_x, test_y = build_net_data(test_data, config)
    # train
    config.n_vocab = len(vocab)
    model = x.MyModel(config)

    model.build(input_shape=(None, config.max_len))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(dev_x, dev_y),
        batch_size=512,
        epochs=1
    )
    model.save_weights(config.save_path)
