#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
import numpy as np


class Config(object):
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/dataset/train.txt'
        self.dev_path = dataset + '/dataset/dev.txt'
        self.text_path = dataset + '/dataset/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/dataset/class.txt').readlines()]
        self.vocab_path = dataset + '/embedding/vocab.pkl'
        self.save_path = './results/' + self.model_name + '.h5'
        self.log_path = '../log/' + self.model_name

        self.embedding_matrix = np.load(dataset + '/embedding/' + embedding)['embeddings'].astype('float32') if embedding != 'random' else None

        self.dropout = 0.5
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.max_len = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_matrix.shape[1] if self.embedding_matrix is not None else 300
        self.hidden_size = 128
        self.num_layers = 2


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.embedding = Embedding(
            input_dim=self.config.embedding_matrix.shape[0],
            output_dim=self.config.embedding_matrix.shape[1],
            input_length=self.config.max_len,
            weights=[self.config.embedding_matrix],
            trainable=False
        )
        self.biRNN = LSTM(
            units=self.config.hidden_size,
            return_sequences=True,
            activation='relu'
        )
        self.dropout = Dropout(self.config.dropout)
        self.flatten = Flatten()
        self.out_put = Dense(units=config.num_classes, activation='softmax')

    def build(self, input_shape):
        super(MyModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.biRNN(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.out_put(x)
        return x
