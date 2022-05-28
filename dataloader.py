#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/03/13
@description:
"""
import json
import os
from typing import Optional, Any
from torchtext.legacy import data
import pytorch_lightning as pl
import torch
from utils import common

class TextDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", batch_size=128, max_length=200):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_iter, self.val_iter, self.test_iter = None, None, None
        self.token2index, self.tag2index, self.index2token, self.index2tag = {}, {}, {}, {}
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # 文本内容，使用自定义的分词方法，将内容转换为小写，设置最大长度等
        REVIEW = data.Field(tokenize=common.char_seg,
                            lower=True,
                            fix_length=self.max_length,
                            batch_first=True)
        # 文本对应的标签
        POLARITY = data.LabelField(dtype=torch.long)
        fields = [(None, None), ("polarity", POLARITY), ("review", REVIEW)]
        train_set, val_set, test_set = data.TabularDataset.splits(
            path=self.data_dir,
            train="train.tsv",
            validation="dev.tsv",
            test="test.tsv",
            format="tsv",
            fields=fields,
            skip_header=True,
        )
        REVIEW.build_vocab(train_set.review, val_set.review, test_set.review)
        POLARITY.build_vocab(train_set.polarity)
        self.token2index = dict(REVIEW.vocab.stoi)
        self.index2token = {key: value for key, value in self.token2index.items()}
        self.tag2index = dict(POLARITY.vocab.stoi)
        self.index2tag = {index: value for index, value in enumerate(POLARITY.vocab.itos)}

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train_set, val_set, test_set),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.review),
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x = batch.review
        y = batch.polarity
        return x, y

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def test_dataloader(self):
        return self.test_iter

    def save_dict(self, save_dir):
        with open(os.path.join(save_dir, "token2index.txt"), 'w', encoding='utf8') as writer:
            writer.write(json.dumps(self.token2index, ensure_ascii=False))

        with open(os.path.join(save_dir, "index2tag.txt"), 'w', encoding='utf8') as writer:
            writer.write(json.dumps(self.index2tag, ensure_ascii=False))
