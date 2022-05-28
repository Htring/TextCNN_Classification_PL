#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: common.py
@time:2022/03/13
@description:
"""
import jieba


def char_seg(content: str):
    """
    基于字符的切分方式
    :param content: 待切分的内容
    :return: list[str]
    """
    return list(content)


def word_seg(content: str):
    """
    基于jieba分词的切分方式
    :param content: 待切分内容
    :return: list[str]
    """
    return jieba.lcut(content)
