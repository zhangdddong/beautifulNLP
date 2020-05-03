#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from model import RNN
import os
import time
import datetime

from data_loader import load_data_and_labels
from data_utils import batch_iter


flags = tf.flags
FLAGS = flags.FLAGS

# 加载数据参数
flags.DEFINE_string('pos_dir', './data/rt-polarity.pos', '正例文本路径')
flags.DEFINE_string('neg_dir', './data/rt-polarity.neg', '负例文本路径')
flags.DEFINE_float('dev_sample_percentage', 0.1, '验证集所占比例')
flags.DEFINE_integer('max_sentence_length', 100, '最大句子长度')

# 模型超参数
flags.DEFINE_string('cell_type', 'vanilla', 'vanilla or lstm or gru')
flags.DEFINE_string('word2vec', None, '词向量预训练模型')
flags.DEFINE_integer('embedding_dim', 300, '字向量维度')
flags.DEFINE_integer('hidden_size', 128, '神经网络隐藏层维度')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'dropout')
flags.DEFINE_float('l2_reg_lambda', 3.0, 'l2正则')

# 模型训练参数
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_integer('num_epochs', 100, 'training epochs')
flags.DEFINE_integer('display_every', 10, '训练输出信息步数')
flags.DEFINE_integer('evaluate_every', 100, '验证模型的训练步数')
flags.DEFINE_integer('checkpoint_every', 100, '存储模型的步数')
flags.DEFINE_integer('num_checkpoints', 5, 'checkpoint存储模型的步数')
flags.DEFINE_float('learning_rate', 1e-3, '学习率')

# 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS.flag_values_dict()
print('\nParameters: ')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr, value))
print("")


def train():
    with tf.device('/cpu:0'):
        x_text, y = load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    text_vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print('Text vocabulary size: {:d}'.format(len(text_vocab_processor.vocabulary_)))
    print('x = {0}'.format(x.shape))
    print('y = {0}'.format(y.shape))
    print('')

    # shuffle数据
    np.random.seed(10)
    shuffle_index = np.random.permutation(np.arange(len(y)))
    x_shuffle = x[shuffle_index]
    y_shuffle = y[shuffle_index]

    # 分割训练集和验证集
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffle[: dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[: dev_sample_index], y_shuffle[dev_sample_index:]
    print('Train/Dev split: {:d}/{:d}\n'.format(len(y_train), len(y_dev)))

    # 训练模型
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(text_vocab_processor.vocabulary_),
                embedded_dim=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_dim=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # 定义训练程序
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)

            # 模型输出路径
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
            print('writing to {}\n'.format(out_dir))

            # 损失和准确率summary
            loss_summary = tf.summary.scalar('loss', rnn.loss)
            acc_summary = tf.summary.scalar('accuracy', rnn.accuracy)

            # 训练summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # 验证summary
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint目录，需要先创建他
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # 存储单词表
            text_vocab_processor.save(os.path.join(out_dir, 'text_vocab'))

            # 初始化所有变量
            sess.run(tf.global_variables_initializer())

            # 预训练word2vec
            if FLAGS.word2vec:
                # 初始化矩阵
                initW = np.random.uniform(-0.25, 0.25, len(text_vocab_processor.vocabulary_), FLAGS.embedding_dim)
                # 加载词向量
                print('Load word2vec file {0}'.format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            # 生成batches
            batches = batch_iter(
                data=list(zip(x_train, y_train)),
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs
            )

            for batch in batches:
                x_batch, y_batch = zip(*batch)

                # 训练
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                    feed_dict
                )
                train_summary_writer.add_summary(summaries, step)

                # 训练日志显示
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # 评估
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rnn.input_text: x_dev,
                        rnn.input_y: y_dev,
                        rnn.dropout_keep_prob: 1.0
                    }
                    summaries_dev, loss, accuracy = sess.run(
                        [dev_summary_op, rnn.loss, rnn.accuracy], feed_dict_dev)
                    dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
