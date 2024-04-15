#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:shi
# software: PyCharm
import torch
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    训练数据集的DataLoader实例
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model——用于训练模型的三元组
    params:		Parameters for the experiments——实验参数

    Returns
    -------
    A training Dataset class instance used by DataLoader——DataLoader使用的训练数据集类实例
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label, sub_samp = torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
        # 指定 label 维度
        trp_label = self.get_label(label)
        # 标签平滑
        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        return triple, trp_label, None, None

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def get_label(self, label):
        # 加载批次训练数据时，将 label 指定维度输出
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def get_neg_ent(self, triple, label):
        def get(triple, label):
            pos_obj = label
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0
            neg_ent = np.int32(
                np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
            neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent


class TestDataset(Dataset):
    """
    验证数据集的DataLoader实例
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model——用于验证模型的三元组
    params:		Parameters for the experiments——实验参数

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation——DataLoader使用的训练数据集类实例
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)