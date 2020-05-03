#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# __author__ = 'zd'
import tensorflow as tf
from tensorflow.contrib import learn
import os
import numpy as np

from data_loader import load_data_and_labels
from data_utils import batch_iter


flags = tf.flags
FLAGS = flags.FLAGS

# 数据集参数
flags.DEFINE_string('pos_dir', 'data/rt-polarity.pos', '正例数据集')
flags.DEFINE_string('neg_dir', 'data/rt-polarity.neg', '负例数据集')

# 验证参数
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_string('checkpoint_dir', './runs/1585013556/checkpoints', '训练的checkpoint路径')

# 评估参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def eval():
    with tf.device('/cpu:0'):
        x_text, y = load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    # 构建词典
    text_path = os.path.join(FLAGS.checkpoint_dir, '..', 'text_vocab')
    text_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_config)
        with sess.as_default():
            # 加载存储的图和变量
            saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

            # 通过name从图中获取placeholder
            input_text = graph.get_operation_by_name('input_text').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

            # 计划评估的Tensor
            predictions = graph.get_operation_by_name('output/predictions').outputs[0]

            # 生成batch
            batches = batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # 收集预测内容
            all_predictions = []
            for x_batch in batches:
                batch_pridictions = sess.run(
                    predictions,
                    {input_text: x_batch, dropout_keep_prob: 1.0}
                )
                all_predictions = np.concatenate([all_predictions, batch_pridictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))


def main(_):
    eval()


if __name__ == '__main__':
    tf.app.run()
