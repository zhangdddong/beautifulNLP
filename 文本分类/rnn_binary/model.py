#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class RNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedded_dim,
                 cell_type, hidden_dim, l2_reg_lambda=0.0):
        """
        :param sequence_length: 
        :param num_classes: 
        :param vocab_size:
        :param embedded_dim:
        :param cell_type: 
        :param hidden_dim:
        :param l2_reg_lambda: 
        """
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedded_dim = embedded_dim
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.l2_reg_lambda = l2_reg_lambda
        self.word_lookup = None

        # 占位符
        self.input_text = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.l2_loss = tf.constant(0.0)
        self.text_length = self._length(self.input_text)

        # embedding layer
        embedded = self.embedding_layer(self.input_text)

        # rnn layer
        rnn_output = self.rnn_layer(embedded, self.dropout_keep_prob)

        # project layer
        logits = self.project_layer(rnn_output)

        # calculate mean cross-entropy loss
        losses = self.loss_layer(logits)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        # predictions
        self.predictions = tf.argmax(logits, axis=1, name='predictions')

        # accuracy
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def embedding_layer(self, input_text, name=None):
        """
        embedding层
        :param input_text: 输入的文本 [batch_size, sentence_length] one-hot encoding ?
        :param name: name
        :return: [batch_size, sentence_length, embedding_dim]
        """
        with tf.device('/cpu:0'), tf.variable_scope('text_embedding' if not name else name):
            self.word_lookup = tf.get_variable(
                name='word_embedding',
                shape=[self.vocab_size, self.embedded_dim],
                initializer=tf.initializers.random_uniform(minval=-1.0, maxval=1.0)
            )
            embedding = tf.nn.embedding_lookup(self.word_lookup, input_text)
            return embedding

    def rnn_layer(self, embedded, dropout_keep_prob, name=None):
        """
        :param embedded: [batch_size, sentence_length, embedding_dim]
        :param dropout_keep_prob: dropout
        :param name: name
        :return: [batch_size, hidden_dim]
        """
        with tf.variable_scope('rnn_layer' if not name else name):
            cell = self._get_cell(self.hidden_dim, self.cell_type)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
            # [batch_size, sentence_length, rnn_dim]
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=embedded,
                sequence_length=self.text_length,
                dtype=tf.float32
            )
            rnn_outputs = self.last_relevant(rnn_outputs, self.text_length)
            return rnn_outputs

    def project_layer(self, rnn_outputs, name=None):
        """
        :param rnn_outputs: [batch_size, hidden_dim]
        :param name:
        :return: [batch_size, num_classes] --> self.input_y
        """
        with tf.variable_scope('logits' if not name else name):
            W = tf.get_variable(
                name='W',
                shape=[self.hidden_dim, self.num_classes],
                initializer=initializers.xavier_initializer()
            )
            b = tf.get_variable(
                name='b',
                shape=[self.num_classes],
                initializer=tf.zeros_initializer()
            )
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            logits = tf.nn.xw_plus_b(rnn_outputs, W, b, name='logits')
            return logits

    def loss_layer(self, logits, name=None):
        """
        :param logits:
        :param name:
        :return:
        """
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            return loss

    @staticmethod
    def _length(seq):
        """
        :param seq: [batch_size, sequence_length]
        :return: list [batch_size] 输入数据的每句话的长度
        """
        mask = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(mask, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == 'vanilla':
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print('Error: ' + cell_type + '!')
            return None

    @staticmethod
    def last_relevant(rnn_outputs, length):
        """
        每句话的最后一个单词的向量
        :param rnn_outputs: [batch_size, sentence_length, hidden_dim]
        :param length: [batch_size]
        :return: [batch_size, hidden_dim] --> 每句话的最后一个单词的向量，代表这个句子
        """
        batch_size = tf.shape(rnn_outputs)[0]
        max_length = int(rnn_outputs.get_shape()[1])
        hidden_dim = int(rnn_outputs.get_shape()[2])
        flat = tf.reshape(rnn_outputs, [-1, hidden_dim])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        return tf.gather(flat, index)
