#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'zd'

import os
import codecs
import tensorflow as tf

from bert_base.train.data_loader import NerProcessor
from bert_base.bert import modeling, tokenization
from bert_base.train.train_helper import set_logging
from bert_base.train.model_utils import filed_based_convert_examples_to_features
from bert_base.train.model_utils import file_based_input_fn_builder
from bert_base.train.model_utils import create_model
from bert_base.bert import optimization

logger = set_logging('NER Training')


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, FLAGS):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param FLAGS:
    :return:
    """
    def model_fn(features, labels, mode, params):
        """
        :param features:
        :param labels:
        :param mode:
        :param params:
        :return:
        """
        logger.info('*** Features ***')
        for name in sorted(features.keys()):
            logger.info(' name = %s, shape = %s' % (name, features[name].shape))
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']

        print('shape of input ids', input_ids.shape)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构造模型，input_idx就是输入的样本idx表示，label_ids就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=label_ids,
            num_labels=num_labels,
            use_one_hot_embeddings=False,
            dropout_rate=FLAGS.dropout_rate,
            lstm_size=FLAGS.lstm_size,
            cell=FLAGS.cell,
            num_layers=FLAGS.num_layers
        )

        """
        tf.trainable_variables(): 返回需要训练的变量列表
        tf.all_variables(): 返回的是所有变量的列表
        """
        tvars = tf.trainable_variables()

        # 加载Bert模型
        if init_checkpoint:
            (assigment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars=tvars,
                init_checkpoint=init_checkpoint
            )
            tf.train.init_from_checkpoint(init_checkpoint, assigment_map)

        output_spec = None
        # 分三种情况，mode分别为训练、验证、测试
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss=total_loss,
                init_lr=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=False
            )
            hook_dict = dict()
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict,
                every_n_iter=FLAGS.save_summary_steps
            )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook]
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, pred_ids):
                return {
                    'eval_loss': tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids)
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


def get_last_checkpoint(model_path):
    """
    用于最终模型应用
    :param model_path:
    :return:
    """
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits: '.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2: -1]
                break
    return last


def train(FLAGS):
    print(FLAGS.bert_config_file)

    processors = {
        'ner': NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # max_seq_length必须小于设置的max_position_embeddings，max_position_embeddings这里是512
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            'Cannot use sequence length %d because the BERT model'
            'was not trained up to sequence length %d' %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )

    # 检查output目录是否存在
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    processor = processors['ner'](FLAGS.output_dir)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case
    )

    """
    tf.ConfigProto: tensorflow config protocol，tensorflow配置协议
    :param log_device_placement: 如果是True，我们可以看到我们的tensor、op是在哪台设备、哪颗CPU上运行的。如果是Flase就看不到。
    :param inter_op_parallelism_threads: 每个进程可用的为进行阻塞操作节点准备线程池中线程数量，设置为0代表让系统选择合适数值。
    :param intra_op_parallelism_threads: 线程池中线程的数量，如果设置为0代表让系统设置合适的数值。
    :param allow_soft_placement: 这个参数制定是否允许计算的“软分配”。
                                 如果这个参数设置为True，那么一个操作在下列情况下会被放在CPU上运行：
                                     1、操作没有GPU的实现
                                     2、没有已知的GPU
                                     3、需要与来自CPU的reftype输入进行协同定位
    """
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True
    )

    """
    tf.estimator.RunConfig: tensorflow运行配置文件
    :param model_dir: 模型的输出路径
    :param save_summary_steps: 多少步进行可视化更新
    :param save_checkpoints_steps: 多少步进行存储ck文件
    :param session_config: session的配置
    """
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train and FLAGS.do_dev:
        # 加载训练数据
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            1.0 * len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs
        )
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        logger.info('***** Running training *****')

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

    label_list = processor.get_labels()

    # 1、将训练数据转化为TF_Record数据
    train_file = os.path.join(FLAGS.output_dir, 'train.tf_record')
    if not os.path.exists(train_file):
        filed_based_convert_examples_to_features(
            examples=train_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer,
            output_file=train_file,
            output_dir=FLAGS.output_dir
        )

    # 2、读取TF_Record训练数据，转化为batch
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True
    )

    # 1、将验证数据转化为TF_Record数据
    eval_file = os.path.join(FLAGS.output_dir, 'eval.tf_record')
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            examples=eval_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer,
            output_file=eval_file,
            output_dir=FLAGS.output_dir
        )

    # 2、读取TF_Record验证数据，转化为batch
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False
    )

    """
    返回的model_fn是一个函数，其定义了模型、训练、评测方法
    并且使用了钩子参数，加载了Bert模型的参数进行了自己模型的参数初始化过程
    tf新的架构方法，通过定义model_fn函数，定义模型，然后通过EstimatorAPI进行模型的其他工作
    EstimatorAPI就可以控制模型的训练、预测和评估工作等。
    """
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        FLAGS=FLAGS
    )

    params = {
        'batch_size': FLAGS.train_batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config
    )

    # 设置early_stopping，防止过拟合
    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=num_train_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=FLAGS.save_checkpoints_steps
    )

    # 训练
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps,
        hooks=[early_stopping_hook]
    )
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
