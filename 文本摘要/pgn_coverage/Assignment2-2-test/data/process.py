#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-07-30 17:59
# @Description: In User Settings Edit
# @Software : PyCharm
import os
import sys
import pathlib
import json
import jieba

root_path = pathlib.Path(__file__).parent.parent.absolute()


def read_samples(filename):
    """
    读文件中的数据
    :param filename:
    :return:
    """
    samples = []
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path):
    """
    将列表写入文件
    :param samples:
    :param file_path:
    :return:
    """
    with open(file_path, 'w', encoding='UTF-8') as f:
        for line in samples:
            f.write(line)
            f.write('\n')


def partition(samples):
    """
    将列表分成训练集、测试集和验证集
    :param samples: list
    :return:
    """
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count <= 1000:
            test.append(sample)
        elif count <= 6000:
            dev.append(sample)
        else:
            train.append(sample)
    print('train length: ', len(train))
    write_samples(train, os.path.join(root_path, 'files', 'train.txt'))
    write_samples(dev, os.path.join(root_path, 'files', 'dev.txt'))
    write_samples(test, os.path.join(root_path, 'files', 'test.txt'))


def is_chinese(word):
    """
    判断是否为中文，如果是中文，返回true
    :param word: str
    :return: bool
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    samples = set()
    json_path = os.path.join(root_path, 'files', '服饰_50k.json')
    with open(json_path, 'r', encoding='UTF-8') as f:
        json_file = json.load(f)

    for line in json_file.values():
        title = line['title'] + ' '
        kb = dict(line['kb']).items()
        kb_merged = ''
        for key, val in kb:
            kb_merged += key + ' ' + val + ' '
        ocr = ' '.join(list(jieba.cut(line['ocr'])))
        reference = ' '.join(list(jieba.cut(line['reference'])))

        texts = list()
        texts.append(title + ocr + kb_merged)
        for text in texts:
            sample = text + '<sep>' + reference
            samples.add(sample)
    write_samples(samples, os.path.join(root_path, 'files', 'samples.txt'))
    partition(samples)
