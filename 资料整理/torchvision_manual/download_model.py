#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-07-19 10:40
# @Description: In User Settings Edit
# @Software : PyCharm
from urllib.parse import urlparse
import torch.utils.model_zoo as model_zoo
import re
import os


def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)

    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)

    model_zoo._download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    return filename


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
path = 'D:/Software/DataSet/models/vgg'
if not (os.path.exists(path)):
    os.makedirs(path)
for url in model_urls.values():
    download_model(url, path)
