#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.crf import crf_log_likelihood
import numpy as np
from tensorflow.contrib.crf import viterbi_decode
import data_utils


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config['lr']
        self.word_dim = config['word_dim']
        self.lstm_dim = config['lstm_dim']
        self.seg_dim = config['seg_dim']
        self.num_tags = config['num_tags']
        self.num_words = config['num_words']
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # 申请占位符
        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='wordInputs')
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segInputs')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')

        used = tf.sign(tf.abs(self.word_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.word_inputs)[0]
        self.sentence_length = tf.shape(self.word_inputs)[-1]

        # embedding层单词和分词信息
        embedding = self.embedding_layer(self.word_inputs, self.seg_inputs, config)

        # bilstm输入层
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bilstm输出层
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # 投影层
        self.logits = self.project_layer(lstm_outputs)

        # 损失层
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope('optimizer'):
            optimizer = self.config['optimizer']
            if optimizer == 'sgd':
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == 'adam':
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == 'adgrad':
                self.opt = tf.train.AdagradDAOptimizer(self.lr)
            else:
                raise Exception('优化器错误')
            grad_vars = self.opt.compute_gradients(self.loss)
            capped_grad_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']),v] for g, v in grad_vars]
            self.train_op = self.opt.apply_gradients(capped_grad_vars, self.global_step)

            # 保存模型
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, word_inputs, seg_inputs, config, name=None):
        """
        :param word_inputs: [batch_size, sentence_length] one-hot encoding
        :param seg_inputs: segment information
        :param config: config
        :param name: the name of layers
        :return: [batch_size, sentence_length, word_embedding + seg_embedding]
        """
        embedding = []
        with tf.variable_scope('word_embedding' if not name else name):
            self.word_lookup = tf.get_variable(
                name='word_embedding',
                shape=[self.num_words, self.word_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.word_lookup, word_inputs))

            if config['seg_dim']:
                with tf.variable_scope('seg_embedding'):
                    self.seg_lookup = tf.get_variable(
                        name='seg_embedding',
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer
                    )
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
        return tf.concat(embedding, axis=-1)

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, sentences_length, emd_size]
        :param lstm_dim:
        :param lengths:
        :param name:
        :return: [batch_size, sentence_length, 2 * lstm_dim]
        """
        with tf.variable_scope('word_biLSTM' if not name else name):
            lstm_cell = {}
            for direction in ['forward', 'backward']:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True
                    )
            outputs, final_status = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell['forward'],
                lstm_cell['backward'],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths
            )
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, sentence_length, 2 * lstm_dim]
        :param name:
        :return: [batch_size, sentence_length, num_tags]
        """
        with tf.variable_scope('project_layer' if not name else name):
            with tf.variable_scope('hidden_layer'):
                W = tf.get_variable(
                    'W',
                    shape=[self.lstm_dim * 2, self.lstm_dim],
                    dtype=tf.float32,
                    initializer=self.initializer
                )
                b = tf.get_variable(
                    'b',
                    shape=[self.lstm_dim],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                out_put = tf.reshape(lstm_outputs, shape=[-1, 2 * self.lstm_dim])
                hidden = tf.tanh(tf.nn.xw_plus_b(out_put, W, b))
            with tf.variable_scope('logits'):
                W = tf.get_variable(
                    'W',
                    shape=[self.lstm_dim, self.num_tags],
                    dtype=tf.float32,
                    initializer=self.initializer
                )
                b = tf.get_variable(
                    'b',
                    shape=[self.num_tags],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                pred = tf.nn.xw_plus_b(hidden, W, b)
        return tf.reshape(pred, [-1, self.sentence_length, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        :param project_logits: [batch_size, sentences_length, num_tags]
        :param lengths: [...]
        :param name:
        :return: scalar loss
        """
        with tf.variable_scope('loss' if not name else name):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=project_logits,
                                                                    labels=self.targets)
            mask = tf.sequence_mask(self.lengths)
            losses = tf.boolean_mask(losses, mask)
        return tf.reduce_mean(losses)

    def decode(self, logits, lengths):
        """
        :param logits: [batch_size, sentences_length, num_tags]
        :param lengths:
        :param matrix:
        :return:
        """
        paths = []
        labels_softmax = np.argmax(logits, axis=-1)
        label_list = labels_softmax.astype(np.int32)
        label_list = label_list.tolist()
        for i, label in enumerate(label_list):
            path = label[:lengths[i]]
            paths.append(path)
        return paths

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train:
        :param batch:
        :return:
        """
        _, words, segs, tags = batch
        feed_dict = {
            self.word_inputs: np.asarray(words),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config['dropout_keep']
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess:
        :param is_train:
        :param batch:
        :return:
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _= sess.run(
                [self.global_step, self.loss, self.train_op], feed_dict
            )
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess:
        :param data_manager:
        :param id_to_tag:
        :return:
        """
        results = []
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, logits = self.run_step(sess, False, batch)
            batch_paths = self.decode(logits, lengths)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results


