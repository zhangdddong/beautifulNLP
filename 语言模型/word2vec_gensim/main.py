#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import multiprocessing
import joblib
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


input_file = './data/data.utf8'
output_file = './data/word2vec.vector'


# 训练模型
model = Word2Vec(LineSentence(input_file), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
joblib.dump(model, output_file)


# 加载并使用，返回numpy.ndarray
m = joblib.load('./data/word2vec.vector')
print(type(m['数']))
