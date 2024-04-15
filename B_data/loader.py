#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:shi
# software: PyCharm
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from collections import defaultdict as ddict
from B_data import TrainDataset, TestDataset


class LoadData(object):
    """
    读取原始三元组并将其转换为标准格式
    Reading in raw triples and converts it into a standard format.

    Parameters
    ----------
    self.p.dataset:         Takes in the name of the dataset (FB15k-237)——采用数据集的名称（默认FB15k-237）

    Returns
    -------
    self.ent2id:            Entity to unique identifier mapping——实体到id的映射
    self.rel2id:            Relation to unique identifier mapping——关系到id的映射
    self.id2ent:            Inverse mapping of self.ent2id——实体到id的逆映射
    self.id2rel:            Inverse mapping of self.rel2id——关系到id的逆映射
    self.num_ent:           Number of entities in the Knowledge graph——知识图谱实体数
    self.num_rel:           Number of relations in the Knowledge graph——知识图谱关系数
    self.embed_dim:         Embedding dimension used——使用的Embedding维度
    self.data['train']:     Stores the triples corresponding to training dataset——训练集三元组(id)
    self.data['valid']:     Stores the triples corresponding to validation dataset——验证集三元组(id)
    self.data['test']:      Stores the triples corresponding to test dataset——测试集三元组(id)
    self.data_iter:	     	The dataloader for different data splits——数据迭代器(id)
    self.triples:           Stores the triples corresponding to all dataset——数据集三元组(id+label)
    """

    def __init__(self, params, device):
        """
        构造函数
        Constructor of the LoadData class

        Parameters
        ----------
        params：模型超参数列表
        device：所用设备

        Returns
        -------
        """
        self.p = params
        self.device = device

        # 创建实体及关系的有序集合
        ent_set, rel_set = OrderedSet(), OrderedSet()
        # 提取所有数据——train、test、valid，构建id与之对应
        for split in ['train', 'test', 'valid']:
            for line in open(self.p.dataset_dir + '{}/{}.txt'.format(self.p.dataset, split)):
                # 删除该行字符串开头和结尾的空格，并以空格符‘\t’分割字符串
                # subject（主语），predicate（谓语），object（宾语）
                # <实体1，关系，实体2>
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
        # 字典：实体2id
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        # 字典：关系2id
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        # 字典的更新——关系反转：关系2id
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        # 字典：id2实体
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        # 字典：id2关系
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        # 实体数
        self.p.num_ent = len(self.ent2id)
        # 关系数
        self.p.num_rel = len(self.rel2id) // 2
        # 输入得分函数的嵌入维度
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        # 创建train、test、valid数据的多级字典——列表
        self.data = ddict(list)
        # 多级字典：实体关系2实体——set
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open(self.p.dataset_dir + '{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                # 实体及关系转为id
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                # 根据数据集类型，加入多级字典
                self.data[split].append((sub, rel, obj))

                # train数据集，需加入多级字典——实体关系2实体
                if split == 'train':
                    # 正向
                    sr2o[(sub, rel)].add(obj)
                    # 反向
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # train、test、valid数据的多级字典转为字典
        self.data = dict(self.data)

        # train数据集，实体关系2实体的多级字典转为字典
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        # test、valid数据集，加入实体关系2实体的多级字典
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # train、test、valid数据集，实体关系2实体的多级字典转为字典
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        # 创建train、test、valid数据三元组的多级字典——含label
        self.triples = ddict(list)

        # train数据集，实体关系2实体的多级字典构建三元组——含label
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        # test、valid数据集，实体关系2实体的多级字典构建三元组——含label
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        # train、test、valid数据三元组的多级字典转为字典——含label
        self.triples = dict(self.triples)

        # 数据集迭代器——各数据集的含label三元组
        self.data_iter = {
            'train': self.get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': self.get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': self.get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': self.get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': self.get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }
        # 获取邻接矩阵，及邻接边类型
        self.edge_index, self.edge_type = self.construct_adj()

    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        """
        获取数据集迭代器
        Get dataset iterator

        Parameters
        ----------

        Returns
        -------
        DataLoader 数据集迭代器

        """
        return DataLoader(
            dataset_class(self.triples[split], self.p),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, self.p.num_workers),
            collate_fn=dataset_class.collate_fn
        )

    def construct_adj(self):
        """
        构造GCN的邻接矩阵
        Constructs the adjacency matrix for GCN

        Parameters
        ----------

        Returns
        -------
        edge_index 邻接矩阵
        edge_type  邻接边类型
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type
