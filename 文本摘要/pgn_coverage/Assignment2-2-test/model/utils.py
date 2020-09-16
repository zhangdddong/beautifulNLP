#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-26 9:37
# @Description: In User Settings Edit
# @Software : PyCharm
import numpy as np
import time
import heapq
import random
import sys
import pathlib
import torch

root_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(root_path)
from model import config


def timer(module):
    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print('{} secs used for {}'.format(cost_time, module))
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def sort_batch_by_len(data_batch):
    res = {
        'x': [],
        'y': [],
        'x_len': [],
        'y_len': [],
        'OOV': [],
        'len_OOV': []
    }
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {
        name: [_tensor[i] for i in sorted_indices] for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
    映射id到字符，包括OOVs中临时id。
    :param id_list: id列表
    :param source_oovs: 临时oov的字符列表
    :param vocab:
    :return:
    """
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]    # might be [UNK]
        except IndexError:
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)


def source2id(source_words, vocab):
    """
    将源文转化成id
    :param source_words:
    :param vocab:
    :return:
    """
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, source_oovs):
    """
    将摘要(reference)映射成id，OOV的词也会被保留。
    :param abstract_words:
    :param vocab:
    :param source_oovs:
    :return:
    """
    ids = []
    unk_id = vocab.UNK
    vocab_size = vocab.size()

    for word in abstract_words:
        i = vocab[word]
        if i == unk_id:
            ids.append(vocab_size + source_oovs.index(word) if word in source_oovs else unk_id)
        else:
            ids.append(i)

    return ids
