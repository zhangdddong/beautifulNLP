#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-08-19 9:21
# @Description: In User Settings Edit
# @Software : PyCharm
import os
import sys
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader

root_path = pathlib.Path(__file__).parent.parent
sys.path.append(root_path)
from model.dataset import collate_fn
from model import config


def evaluate(model, val_data, epoch):
    """
    评估模型
    :param model: (torch.nn.Module) The model to evaluate.
    :param val_data: (dataset.PairDataset) The evaluation data set.
    :param epoch: (int) The epoch number.
    :return:
    """
    print('validating')
    val_loss = []
    with torch.no_grad():
        DEVICE = config.DEVICE
        val_dataloader = DataLoader(
            dataset=val_data,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        for batch, data in enumerate(val_dataloader):
            x, y, x_len, y_len, oov, len_oovs = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x_len = x_len.to(DEVICE)
            len_oovs = len_oovs.to(DEVICE)
            loss = model(
                x,
                x_len,
                y,
                len_oovs,
                batch=batch,
                num_batches=len(val_dataloader)
            )
            val_loss.append(loss.item())
    return np.mean(val_loss)
