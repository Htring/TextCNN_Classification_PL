#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainner.py
@time:2022/03/13
@description:
"""
import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model import TextCNNPl
from dataloader import TextDataModule
from utils.common import char_seg


def training(param):
    dm = TextDataModule(batch_size=param.batch_size)
    checkpoint_callback = ModelCheckpoint(monitor='f1_score',
                                          filename="textcnn-{epoch:03d}-{val_loss:.2f}-{f1_score:.2f}",
                                          dirpath=param.save_dir,
                                          save_top_k=3)

    args.vocab_size = len(dm.token2index)
    args.output_size = len(dm.tag2index)
    model = TextCNNPl(args)
    if param.load_pre:
        model = model.load_from_checkpoint(param.pre_ckpt_path)
    logger = TensorBoardLogger("log_dir", name="textcnn_pl")

    trainer = pl.Trainer(logger=logger, gpus=1,
                         callbacks=[checkpoint_callback],
                         max_epochs=param.epoch,
                         precision=16)
    trainer.fit(model=model, datamodule=dm)
    dm.save_dict(param.save_dir)
    trainer.test(model, dm)
    model.to_torchscript(os.path.join(param.save_dir, "best_model.pt"))

def model_use(param):

    def _load_dict(dir_name):
        with open(os.path.join(dir_name, 'token2index.txt'), 'r', encoding='utf8') as reader:
            token2index = json.load(reader)

        with open(os.path.join(dir_name, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            index2tag = json.load(reader)

        return token2index, index2tag

    def _number_data(content):
        number_data = []
        for char in char_seg(content):
            number_data.append(token2index.get(char, token2index.get("<unk>")))
        if len(number_data) < param.max_length:
            number_data.extend([token2index.get("<pad>")] * (param.max_length - len(number_data)))
        else:
            number_data = number_data[: param.max_length]
        return torch.tensor([number_data], dtype=torch.long)


    token2index, index2tag = _load_dict(param.save_dir)
    param.vocab_size = len(token2index)
    param.output_size = len(index2tag)
    model = TextCNNPl.load_from_checkpoint(param.pre_ckpt_path, param=param)
    test_data = "空间大，上路很有面子"
    result_index = model.forward(_number_data(test_data)).argmax(dim=-1)[0].item()
    print(index2tag.get(str(result_index)))



if __name__ == '__main__':
    model_parser = TextCNNPl.add_argparse_args()
    parser = ArgumentParser(parents=[model_parser])
    parser.add_argument('-lr', type=float, default=0.01, help='学习率')
    parser.add_argument('-batch_size', type=int, default=300, help='批次数据大小')
    parser.add_argument('-epoch', type=int, default=15)
    parser.add_argument('-embedding_dim', type=int, default=128, help='词向量的维度')
    parser.add_argument('-save_dir', type=str, default="model_save/textcnn", help='模型存储位置')
    parser.add_argument('-load_pre', type=bool, default=False, help='是否加载已经训练好的ckpt')
    parser.add_argument('-pre_ckpt_path', type=str,
                        default="model_save/textcnn/textcnn-epoch=006-val_loss=0.19-f1_score=0.94.ckpt",
                        help='是否加载已经训练好的ckpt')

    args = parser.parse_args()
    training(args)
    # model_use(args)
