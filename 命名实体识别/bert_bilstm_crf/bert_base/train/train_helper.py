#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import os
import logging


class NTLogger(object):
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s-%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s-%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s-%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s-%s' % (self.context, msg), flush=True)


def set_logging(context, verbose=False):
    # nt是windows系统，posix是linux系统。如果是nt则直接返回
    if os.name == 'nt':
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logging.handlers = []
    logger.addHandler(console_handler)
    return logger

