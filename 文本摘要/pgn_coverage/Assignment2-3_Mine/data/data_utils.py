#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-03 13:41
# @Description: In User Settings Edit
# @Software : PyCharm
import os
import pathlib
import sys
root_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(root_path)


def read_sample(filename):
    samples = []
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path):
    with open(file_path, 'a', encoding='UTF-8') as sf:
        for line in samples:
            sf.write(line + '\n')


def partition(samples):
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count % 1000 == 0:
            print(count)
        if count <= 1000:  # Test set size.
            test.append(sample)
        elif count <= 6000:  # Dev set size.
            dev.append(sample)
        else:
            train.append(sample)
    print('train: ', len(train))

    write_samples(train, '../files/train.txt')
    write_samples(dev, '../files/dev.txt')
    write_samples(test, '../files/test.txt')


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
