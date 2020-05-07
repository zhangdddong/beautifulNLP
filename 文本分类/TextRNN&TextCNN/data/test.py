#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import numpy as np


embedding = np.load('embedding/embedding_SougouNews.npz')["embeddings"].astype('float32')
print(embedding[0].shape)
print(type(embedding))
