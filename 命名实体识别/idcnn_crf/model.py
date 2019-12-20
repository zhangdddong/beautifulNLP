#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
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

        # id_cnn模型参数
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            }
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.word_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        # embedding层单词和分词信息
        embedding = self.embedding_layer(self.word_inputs, self.seg_inputs, config)

        # idcnn输入层
        idcnn_inputs = tf.nn.dropout(embedding, self.dropout)

        # idcnn layer层
        idcnn_outputs = self.IDCNN_layer(idcnn_inputs)

        # 投影层
        self.logits = self.project_layer_idcnn(idcnn_outputs)

        # 损失层
        self.loss = self.crf_loss_layer(self.logits, self.lengths)

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
        with tf.variable_scope('word_embedding' if not name else name), tf.device('/cpu:0'):
            self.word_lookup = tf.get_variable(
                name='word_embedding',
                shape=[self.num_words, self.word_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.word_lookup, word_inputs))

            if config['seg_dim']:
                with tf.variable_scope('seg_embedding'), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name='seg_embedding',
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer
                    )
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
        return tf.concat(embedding, axis=-1)

    def IDCNN_layer(self, idcnn_inputs, name = None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :param name:
        :return: [batch_size, num_steps, cnn_output_withd]
        """
        idcnn_inputs = tf.expand_dims(idcnn_inputs, 1)
        reuse = False
        if not self.config['is_train']:
            reuse = True
        with tf.variable_scope('idcnn' if not name else name):
            shape = [1, self.filter_width, self.embedding_dim, self.num_filter]

            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                initializer=self.initializer
            )

            layer_input = tf.nn.conv2d(
                idcnn_inputs,
                filter_weights,
                strides=[1,1,1,1],
                padding='SAME',
                name='init_layer'
            )

            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope('conv-layer-%d' % i, reuse = tf.AUTO_REUSE):
                        w = tf.get_variable(
                            'fliter_w',
                            shape=[1, self.filter_width, self.num_filter, self.num_filter],
                            initializer=self.initializer
                        )

                        b = tf.get_variable('filterB', shape=[self.num_filter])

                        conv = tf.nn.atrous_conv2d(
                            layer_input,
                            w,
                            rate=dilation,
                            padding="SAME"
                        )

                        conv  = tf.nn.bias_add(conv, b)

                        conv = tf.nn.relu(conv)

                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim  =  totalWidthForLastDim + self.num_filter
                        layer_input  = conv

            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width  = totalWidthForLastDim
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param idcnn_outputs: [batch_size, num_steps, emb_size]
        :param name:
        :return: [batch_size, num_steps, emb_size]
        """
        with tf.variable_scope('idcnn_project' if not name else name):

            with tf.variable_scope('idcnn_logits'):
                W = tf.get_variable(
                    "W",
                    shape=[self.cnn_output_width, self.num_tags],
                    dtype=tf.float32,
                    initializer=self.initializer
                )

                b = tf.get_variable(
                    "b",
                    initializer=tf.constant(0.001, shape=[self.num_tags])
                )

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.sentence_length, self.num_tags])

    def crf_loss_layer(self, project_logits, lengths, name=None):
        """
        :param project_logits: [batch_size, sentenes_length, num_tags]
        :param lengths: [...]
        :param name:
        :return: scalar loss
        """
        with tf.variable_scope('crf_loss' if not name else name):
            small_value = -10000.0
            start_logits = tf.concat(
                [
                    small_value * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                    tf.zeros(shape=[self.batch_size, 1, 1])
                ], axis=-1
            )
            pad_logits = tf.cast(
                small_value * tf.ones(shape=[self.batch_size, self.sentence_length, 1]),
                dtype=tf.float32
            )
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [
                    tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32),
                    self.targets
                ], axis=-1
            )
            self.trans = tf.get_variable(
                'transitions',
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer
            )
            log_likelihood, self.trans =  crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1
            )
        return tf.reduce_mean(-log_likelihood)

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, sentences_length, num_tags]
        :param lengths:
        :param matrix:
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
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
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, logits = self.run_step(sess, False, batch)
            batch_paths = self.decode(logits, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results


