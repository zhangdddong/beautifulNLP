#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-03 13:08
# @Description: In User Settings Edit
# @Software : PyCharm
import torch


pointer = True
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False
source = 'big_samples'
data_path = '../files/{}.txt'.format(source)
val_data_path = '../files/dev.txt'
test_data_path = '../files/test.txt'
stop_word_file = '../files/HIT_stop_words.txt'
if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:
            model_name = 'pgn'
else:
    model_name = 'baseline'
encoder_save_name = '../saved_model/' + model_name + '/encoder.pt'
decoder_save_name = '../saved_model/' + model_name + '/decoder.pt'
attention_save_name = '../saved_model/' + model_name + '/attention.pt'
reduce_state_save_name = '../saved_model/' + model_name + '/reduce_state.pt'
losses_path = '../saved_model/' + model_name + '/val_losses.pkl'
log_path = '../runs/' + model_name


hidden_size = 512
dec_hidden_size = 512
embed_size = 512
max_vocab_size = 20000
embed_file = None
max_src_len = 300
max_tgt_len = 100
truncate_src = True
truncate_tgt = True
min_dec_steps = 30
max_dec_steps = 100
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 32
max_grad_norm = 2.0
is_cuda = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1

# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.6
