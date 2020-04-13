#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import numpy as np
from sklearn import metrics


a = [[1, 0, 0], [0, 1, 0]]
b = [[1, 0, 0], [0, 1, 0]]

print(metrics.accuracy_score(a, b))
