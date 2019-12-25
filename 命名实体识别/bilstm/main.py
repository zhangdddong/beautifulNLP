#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import os
import tensorflow as tf
import data_loader
import model_utils
import pickle
import data_utils
import itertools
from model import Model
from data_utils import load_word2vec
import numpy as np
import time


flags = tf.app.flags

# operation parameters
flags.DEFINE_boolean('train', True, 'Are you train?')
flags.DEFINE_boolean('clean', True, 'Are you clean file?')

# config parameters
flags.DEFINE_integer('seg_dim', 20, 'seg embedding dimension')
flags.DEFINE_integer('word_dim', 100, 'word embedding dimension')
flags.DEFINE_integer('lstm_dim', 100, 'num of hidden utils in lstm')
flags.DEFINE_string('tag_schema', 'BIOES', 'encoding method')

# training parameters
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float('dropout', 0.5, 'Dropout tate')
flags.DEFINE_integer('batch_size', 120, 'batch size')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_string('optimizer', 'adam', 'optimizer')
flags.DEFINE_boolean('pre_emb', True, 'Are you train word embedding?')

flags.DEFINE_integer('max_epoch', 100, 'num of training')
flags.DEFINE_integer('setps_chech', 10, 'steps per checkpoint')
flags.DEFINE_string('ckpt_path', 'ckpt', 'the position of model')
flags.DEFINE_string('log_file', 'train.log', 'the log of training')
flags.DEFINE_string('map_file', 'maps.pkl', 'dictionary of tag and word')
flags.DEFINE_string('vocab_file', 'vocab.json', 'word embedding')
flags.DEFINE_string('config_file', 'config_file', 'config file')
flags.DEFINE_string('result_path', 'result', 'the path of result')
flags.DEFINE_string('emb_file', os.path.join('data', 'wiki_100.utf8'), 'the path of word embedding')
flags.DEFINE_string('train_file', os.path.join('data', 'ner.train'), 'the path of training data')
flags.DEFINE_string('dev_file', os.path.join('data', 'ner.dev'), 'the path of dev data')
flags.DEFINE_string('test_file', os.path.join('data', 'ner.test'), 'the path of test data')

FLAGS = tf.app.flags.FLAGS

assert FLAGS.clip < 5.1, 'error'
assert 0 < FLAGS.dropout < 1, 'the dropout between 0 and 1'
assert FLAGS.lr > 0, 'the lr must up 0'
assert FLAGS.optimizer in ['adam', 'sgd', 'adagrad'], 'the optimizer must in [adam sgd adagrad]'


def evaluate(sess, model, name, manager, id_to_tag, logger):
    logger.info('evaluate:{}'.format(name))
    ner_results = model.evaluate(sess, manager, id_to_tag)
    eval_lines = model_utils.test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info('new best dev f1 socre:{:>.3f}'.format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info('new best test f1 score:{:>.3f}'.format(f1))
        return f1 > best_test_f1


def train():
    # 加载数据集
    train_sentences = data_loader.load_sentences(FLAGS.train_file)
    dev_sentences = data_loader.load_sentences(FLAGS.dev_file)
    test_sentences = data_loader.load_sentences(FLAGS.test_file)

    # 转换编码
    data_loader.update_tag_scheme(train_sentences, FLAGS.tag_schema)
    data_loader.update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    data_loader.update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # 创建单词和词典映射
    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            dico_words_train = data_loader.word_mapping(train_sentences)[0]
            dico_word, word_to_id, id_to_word = data_utils.augment_with_pretrained(
                dico_words_train.copy(),
                FLAGS.emb_file,
                list(
                    itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _, word_to_id, id_to_word = data_loader.word_mapping(train_sentences)
        _, tag_to_id, id_to_tag = data_loader.tag_mapping(train_sentences)
        with open(FLAGS.map_file, 'wb') as f:
            pickle.dump([word_to_id, id_to_word, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, 'rb') as f:
            word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

    # 准备数据
    train_data = data_loader.prepare_dataset(train_sentences, word_to_id, tag_to_id)
    dev_data = data_loader.prepare_dataset(dev_sentences, word_to_id, tag_to_id)
    test_data = data_loader.prepare_dataset(test_sentences, word_to_id, tag_to_id)

    # 将数据分批处理
    train_manager = data_utils.BatchManager(train_data, FLAGS.batch_size)
    dev_manager = data_utils.BatchManager(dev_data, FLAGS.batch_size)
    test_manager = data_utils.BatchManager(test_data, FLAGS.batch_size)

    # 创建不存在的文件夹
    model_utils.make_path(FLAGS)

    # 判断配置文件
    if os.path.isfile(FLAGS.config_file):
        config = model_utils.load_config(FLAGS.config_file)
    else:
        config = model_utils.config_model(FLAGS, word_to_id, tag_to_id)
        model_utils.save_config(config, FLAGS.config_file)

    # 配置印logger
    log_path = os.path.join('log', FLAGS.log_file)
    logger = model_utils.get_logger(log_path)
    model_utils.print_config(config, logger)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    step_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = model_utils.create(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_word, logger)
        logger.info('开始训练')
        loss = []
        start = time.time()
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.setps_chech == 0:
                    iteration = step // step_per_epoch + 1
                    logger.info("iteration{}: step{}/{}, NER loss:{:>9.6f}".format(iteration, step % step_per_epoch,
                                                                                   step_per_epoch, np.mean(loss)))
                    loss = []
            best = evaluate(sess, model, 'dev', dev_manager, id_to_tag, logger)

            if best:
                model_utils.save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, 'test', test_manager, id_to_tag, logger)
        t = time.time() - start
        logger.info('cost time: %f' % t)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main)
