#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

from data_load import get_features
from model_utils import get_matrix
from model import viterbi


if __name__ == '__main__':
    file_path = './data/ner.train'

    word2id, id2word, tag2id, id2tag = get_features(file_path)
    pi, A, B = get_matrix(word2id, tag2id, id2tag, file_path)

    x = '北京是中国的首都'
    viterbi(x, pi, A, B, word2id, id2tag, tag2id)