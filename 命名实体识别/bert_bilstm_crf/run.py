#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import tensorflow as tf

from bert_base.train.bert_lstm_ner import train


flags = tf.flags
FLAGS = flags.FLAGS

# 输入输出地址
flags.DEFINE_string('data_dir', 'data', '数据集地址')
flags.DEFINE_string('output_dir', 'output', '输出地址')

# Bert相关参数
flags.DEFINE_string('bert_config_file', 'chinese_L-12_H-768_A-12/bert_config.json', 'Bert配置文件')
flags.DEFINE_string('vocab_file', 'chinese_L-12_H-768_A-12/vocab.txt','vocab_file')
flags.DEFINE_string('init_checkpoint','chinese_L-12_H-768_A-12/bert_model.ckpt', 'init_checkpoint')

# 训练和校验的相关参数
flags.DEFINE_bool('do_train', True, '是否开始训练')
flags.DEFINE_bool('do_dev', True, '是否开始校验')
flags.DEFINE_bool('do_test', True, '是否开始测试')

flags.DEFINE_bool('do_lower_case', True, '是否转换小写')

# 模型相关的
flags.DEFINE_integer('lstm_size', 128, 'lstm_size')
flags.DEFINE_integer('num_layers', 1, 'num_layers')
flags.DEFINE_integer('max_seq_length', 128, 'max_seq_length')
flags.DEFINE_integer('train_batch_size', 64, 'train_batch_size')
flags.DEFINE_integer('dev_batch_size',64, 'dev_batch_size')
flags.DEFINE_integer('test_batch_size', 32, 'test_batch_size')
flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_steps')
flags.DEFINE_integer('iterations_per_loop', 1000, 'iterations_per_loop')
flags.DEFINE_integer('save_summary_steps', 500, 'save_summary_steps')

flags.DEFINE_string('cell', 'lstm', 'cell')

flags.DEFINE_float('learning_rate', 5e-5, 'learning_rate')
flags.DEFINE_float('dropout_rate', 0.5, 'dropout_rate')
flags.DEFINE_float('clip', 0.5, 'clip')
flags.DEFINE_float('num_train_epochs', 10.0, 'num_train_epochs')
flags.DEFINE_float("warmup_proportion", 0.1,'warmup_proportion')    # 慢热比率，学习率慢慢增加到正常值


def train_ner():
    train(FLAGS)


if __name__ == '__main__':
    train_ner()
