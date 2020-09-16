#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# @license : Copyright(C), Your Company 
# @Author: Zhang Dong
# @Contact : 1010396971@qq.com
# @Date: 2020-09-03 15:38
# @Description: In User Settings Edit
# @Software : PyCharm
from googletrans import Translator
translator = Translator(service_urls=['translate.google.cn'])
result = translator.translate('你好', dest='en').text
print(result)
