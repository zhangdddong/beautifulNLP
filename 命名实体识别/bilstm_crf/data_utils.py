#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import jieba
import math
import random
import os
import numpy as np


def check_bio(tags):
    """
    检测输入的标签是否为bio标签
    如果不是bio标签
    那么错误类型为：（1）编码不在BIO中（2）第一个编码是I（3）当前编码不是B，前一个编码不是O
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        tag_list = tag.split('-')
        if len(tag_list) != 2 or tag_list[0] not in ['B', 'I']:
            return False
        if tag_list[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i-1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True


def bio_to_bioes(tags):
    """
    将BIO转换成BIOES，并将新的标签返回回去
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('非法编码')
    return new_tags


def bioes_to_bio(tags):
    """
    BIOES转换为BIO，返回新的标签
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'I-'))
        else:
            raise Exception('非法编码')
    return new_tags


def create_dico(item_list):
    """
    对比item_list中的每一个items
    :param item_list:
    :return:
    """
    assert type(item_list) == list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    创建item_to_id id_to_item，item的排序按照词典中出现的次数
    :param dico:
    :return:
    """
    sorted_item = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_item)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def get_seg_feature(words):
    """
    利用结巴分词
    类是BIOES标注法，0表示单字成词，1表示一个词的开始，2表示一个词的中间，3表示一个词的结束
    :param words:
    :return:
    """
    seg_features = []

    word_list = list(jieba.cut(words))

    for word in word_list:
        if len(word) == 1:
            seg_features.append(0)
        else:
            temp = [2] * len(word)
            temp[0] = 1
            temp[-1] = 3
            seg_features.extend(temp)
    return seg_features


def load_word2vec(emb_file, id_to_word, word_dim, old_weights):
    """
    :param emb_file:
    :param id_to_word:
    :param word_dim:
    :param old_weight:
    :return:
    """
    new_weights = old_weights
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(open(emb_file, encoding='UTF-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('waring: %i initializer lines' % emb_invalid)
    num_words = len(id_to_word)
    for i in range(num_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
    print('加载了 %i 个词向量' % len(pre_trained))

    return new_weights


def augment_with_pretrained(dico_train, emb_path, test_word):
    """
    :param dico_train:
    :param emb_path:
    :param test_word:
    :return:
    """
    assert os.path.isfile(emb_path)

    # 加载词向量
    pretrained = set([line.rsplit()[0].strip() for line in open(emb_path, encoding='UTF-8')])
    if test_word is None:
        for word in pretrained:
            dico_train[word] = 0
    else:
        for word in test_word:
            if any(x in pretrained for x in [word, word.lower()]) and word not in dico_train:
                dico_train[word] = 0
    word_to_id, id_to_word = create_mapping(dico_train)

    return dico_train, word_to_id, id_to_word


class BatchManager(object):
    def __init__(self, data, batch_size):
        self._batch_data = self._sort_and_pad(data, batch_size)
        self._len_data = len(self._batch_data)

    @staticmethod
    def pad_data(data):
        word_list = []
        word_id_list = []
        seg_list = []
        tag_id_list = []
        max_length = max([len(s[0]) for s in data])
        for line in data:
            words, word_ids, segs, tag_ids = line
            padding = [0] * (max_length - len(words))
            word_list.append(words + padding)
            word_id_list.append(word_ids + padding)
            seg_list.append(segs + padding)
            tag_id_list.append(tag_ids + padding)
        return [word_list, word_id_list, seg_list, tag_id_list]

    def _sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size:(i + 1) * batch_size]))
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self._batch_data)
        for i in range(self._len_data):
            yield self._batch_data[i]

