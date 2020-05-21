#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import tensorflow as tf
import tensorflow_addons as tfa


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bilstm_crf'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.vocab_path = dataset + '/data/vocab.pkl'   # 词表
        self.save_path = dataset + '/saved_dict/'       # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name     # 日志存储
        self.map_file = dataset + '/data/map.pkl'       # 字典映射文件
        self.emb_file = dataset + '/data/wiki_100.utf8'     # 外部词向量文件
        self.datasetpkl = dataset + '/data/dataset.pkl'     # 数据存储文件
        self.modeldatasetpkl = dataset + '/data/modeldataset.pkl'    # 模型需要数据文件
        self.embedding_matrix_file = dataset + '/data/word_embedding_matrix.npy'   # 词向量压缩好的文件

        self.embsize = 100      # 词向量维度
        self.pre_emb = True     # 是否需要词嵌入
        self.tags_num = 13      # 标签数量

        self.dropout = 0.5
        self.n_vocab = 0
        self.num_epochs = 100
        self.batch_size = 128
        self.max_len = 200
        self.learning_rate = 1e-3
        self.hidden_size = 128
        self.tag_schema = 'BIOES'


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(config.n_vocab, config.embsize)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                config.hidden_size,
                return_sequences=True
            )
        )
        self.dense = tf.keras.layers.Dense(config.tags_num)
        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(config.tags_num, config.tags_num)),
            trainable=False
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, text, labels=None, training=None):
        # 向量[batch_size]  维度为batch_size
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.biLSTM(inputs)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
