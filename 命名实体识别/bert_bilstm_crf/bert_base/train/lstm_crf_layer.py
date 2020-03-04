#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import tensorflow as tf
import tensorflow.contrib.crf as crf
import tensorflow.contrib.rnn as rnn


class BLSTM_CRF(object):
    def __init__(self, embedd_chars, hidden_unit, cell_type, num_layers, dropout_rate, initializers,
                 num_labels, seq_length, labels, lengths, is_training):
        """
        :param embedd_chars: 嵌入层 [batch_size, sentence_length, embedding]
        :param hidden_unit: LSTM隐含层单元数
        :param cell_type: RNN的类型，LSTM or GRU
        :param num_layers: RNN的层数
        :param dropout_rate:
        :param initializers: 初始化方法
        :param num_labels: 标签的数量
        :param seq_length: 序列的最大长度
        :param labels: 真实标签
        :param lengths: 每个batch下序列的真实长度 [batch]
        :param is_training: 是否训练
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedd_chars = embedd_chars
        self.initializers = initializers
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.labels = labels
        self.lengths = lengths
        self.is_training = is_training
        self.embedding_dims = embedd_chars.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络层
        :param crf_only: 是否使用crf
        :return:
        """
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            lstm_output = self.blstm_layer(self.embedded_chars)    # LSTM层
            logits = self.project_bilstm_layer(lstm_output)

        # CRF层
        loss, trans = self.crf_layer(logits)
        pred_id, _ = crf.crf_decode(
            potentials=logits,
            transition_params=trans,
            sequence_length=self.lengths
        )

        return (loss, logits, trans, pred_id)

    def _with_cell(self):
        """
        获取RNN的类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)

        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._with_cell()
        cell_bw = self._with_cell()
        if self.dropout_rate is not None:
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)

        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """
        :param embedding_chars: [batch_size, sentence_length, embedding]
        :return: [batch_size, sentence_length, 2 * lstm_dimension(lstm_unit)]
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_labels, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)

        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, sentence_length, 2 * lstm_dimension]
        :param name: name
        :return: [batch_size, sentence_length, num_tags]
        """
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                W = tf.get_variable(
                    'W',
                    shape=[self.hidden_unit * 2, self.hidden_unit],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer()
                )
                b = tf.get_variable(
                    'b',
                    shape=[self.hidden_unit],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )

                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            with tf.variable_scope('logits'):
                W = tf.get_variable(
                    'W',
                    shape=[self.hidden_unit, self.num_labels],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer()
                )
                b = tf.get_variable(
                    'b',
                    shape=[self.num_labels],
                    dtype=tf.float32,
                    initializer=self.initializers.xavier_initializer()
                )
                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        :param embedding_chars: [batch_size, sentence_length, embedding]
        :param name: name
        :return: [batch_size, sentence_length, num_tags]
        """
        with tf.variable_scope('logits'):
            W = tf.get_variable(
                'W',
                shape=[self.embedding_dims, self.num_labels],
                dtype=tf.float32,
                initializer=self.initializers.xavier_initializer()
            )
            b = tf.get_variable(
                'b',
                shape=[self.num_labels],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

            # [batch_size, embedding_dims]
            output = tf.reshape(
                embedding_chars,
                shape=[-1, self.embedding_dims]
            )

            pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))

        return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param logits: [batch_size, sentence_length, num_tags]
        :return: loss and trans
        """
        with tf.variable_scope('crf_loss'):
            trans = tf.get_variable(
                'transitions',
                shape=[self.num_lables, self.num_layers],
                initializer=self.initializers.xavier_initializer()
            )

            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths
                )
                return tf.reduce_mean(-log_likelihood), trans

