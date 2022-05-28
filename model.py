#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2022/03/13
@description:
"""
from typing import Optional, Any
import torchmetrics
import torch
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from argparse import ArgumentParser


class TextCNNPl(pl.LightningModule):

    @staticmethod
    def add_argparse_args() -> ArgumentParser:
        parser = ArgumentParser(description='TextCNN text classifier', add_help=False)
        parser.add_argument('-max_length', type=int, default=200, help='textcnn最大句子长度 default 200')
        parser.add_argument('-filter_number', type=int, default=60, help='textcnn输出通道数, default 60')
        parser.add_argument('-kernel_list', type=list, default=[3, 4, 5], help='textcnn卷积核长度, default [3,4,5]')
        parser.add_argument('-drop_out', type=float, default=0.5, help='textcnn dropout层， default 0.5')
        return parser

    def __init__(self, param):
        super().__init__()
        self.lr = param.lr
        self.embedding = nn.Embedding(param.vocab_size, param.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=param.embedding_dim, out_channels=param.filter_number,
                                    kernel_size=kernel),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=param.max_length - kernel + 1))
            for kernel in param.kernel_list
        ])
        self.fc = nn.Linear(param.filter_number * len(param.kernel_list), param.output_size)
        self.dropout = nn.Dropout(param.drop_out)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        pred: torch.Tensor = self.forward(x)
        loss = F.cross_entropy(pred, y)
        pred_index = pred.argmax(dim=-1)
        f1_score = torchmetrics.functional.f1(pred_index, y, average="micro")
        accuracy = torchmetrics.functional.accuracy(pred_index, y, average="micro")
        recall = torchmetrics.functional.recall(pred_index, y, average="micro")
        self.log("val_loss", loss)
        self.log("f1_score", f1_score)
        self.log("recall", recall)
        self.log("accuracy", accuracy)
        return {"true": y, "pred": pred_index, "loss": loss, "f1_score": f1_score}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)

    def _test_unit(self, outputs):
        print()
        pred_lst, true_lst = [], []
        for batch_result in outputs:
            pred_lst.extend(batch_result["pred"].cpu().tolist())
            true_lst.extend(batch_result['true'].cpu().tolist())
        report = classification_report(true_lst, pred_lst)
        print(report)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, y = batch
        pred: torch.Tensor = self.forward(x)
        loss = F.cross_entropy(pred, y)
        pred_index = pred.argmax(dim=-1)
        f1_score = torchmetrics.functional.f1(pred_index, y, average="micro")
        self.log("val_loss", loss)
        self.log("f1_score", f1_score)
        return {"true": y, "pred": pred_index, "loss": loss, "f1_score": f1_score}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x) -> Any:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 维度转换，满足一维卷积的输入
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.fc(out)  # 结果输出[128, 2]
        out = self.dropout(out)  # 构建dropout层out = self.dropout(out)  # 构建dropout层
        return out
