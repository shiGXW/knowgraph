#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import torch
import logging
import numpy as np
from transformers import BertTokenizer, BertModel


class BertBase():
    def __init__(self, model_name, device):
        # 设置输出日志
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # 初始化及检查GPU设备
        if torch.cuda.is_available():
            self.device = torch.device(device)
            logging.info("当前正在使用GPU进行计算.\n")
        else:
            self.device = torch.device("cpu")
            logging.info("当前没有找到可用的GPU，正在使用CPU进行计算.\n")

        # 初始化tokenizer和model，并将其放在GPU上（如果可用）
        # 加载中文 BERT 模型和分词器
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    def train(self, sentences1, sentences2, batch_size=32):
        # 按照每批计算词句向量
        sentence_embeddings1 = self.batch_sentences(sentences1, batch_size)
        sentence_embeddings2 = self.batch_sentences(sentences2, batch_size)

        # 词句向量放入GPU
        sentence_embeddings1 = sentence_embeddings1.to(self.device)
        sentence_embeddings2 = sentence_embeddings2.to(self.device)

        similarities, similarity_indexs = self.compare_sentences(sentence_embeddings1, sentence_embeddings2)

        # 清除缓存
        del sentence_embeddings1, sentence_embeddings2

        return similarities, similarity_indexs

    def batch_sentences(self, sentences, batch_size=32):
        # 初始化存储结果的容器
        sentence_embeddings = None

        # 将句子按照批次拆分
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # 使用tokenizer对批次数据进行处理
            encoded_inputs = self.tokenizer(
                batch,  # 即将被编码的批次句子
                padding=True,  # 自动填充句子为最大长度，默认用 0 填充
                truncation=True,  # 超过max_length指定的限制时截断
                max_length=self.model.config.max_position_embeddings,
                return_tensors="pt"
            )

            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)

            # 获取词向量，在GPU上运行模型
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            # 提取句子向量
            batch_sentence_embeddings = outputs[0][:, 0, :]
            if sentence_embeddings is None:
                sentence_embeddings = batch_sentence_embeddings.cpu()
            else:
                sentence_embeddings = torch.cat([sentence_embeddings, batch_sentence_embeddings.cpu()], 0)
            # 清除缓存
            del encoded_inputs, input_ids, outputs, attention_mask, batch_sentence_embeddings

        # 返回所有批次计算出的相似度结果
        return sentence_embeddings

    def compare_sentences(self, sentence_embeddings1, sentence_embeddings2):
        # 初始化存储结果的容器
        similarities = []
        similarity_indexs = []

        for index in range(0, sentence_embeddings1.shape[0]):
            # 计算余弦相似度
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
            one_embedding = sentence_embeddings1[index].repeat(sentence_embeddings2.shape[0], 1)
            one_similaritie = cos(one_embedding, sentence_embeddings2).cpu().numpy()

            # 将每个句子对的相似度最高的index添加到结果容器
            similarities.append(np.max(one_similaritie))
            similarity_indexs.append(np.argmax(one_similaritie))

            # 清除缓存
            del one_embedding, one_similaritie

            # 返回所有批次计算出的相似度最高的index
        return similarities, similarity_indexs


if __name__ == '__main__':
    # 使用示例
    # 假设有足够多的句子
    sentences1 = ["玻璃纤维", "玻璃纤维", "玻璃纤维", "玻璃纤维", "玻璃纤维", "黄河南大街70号8门", "另一个句子1", "更多句子..."]
    # 与sentences1长度相同
    sentences2 = ["玻璃纤维及其制品", "玻璃纤维纱", "玻璃纤维布", "玻璃纤维毡", "玻璃纤维带", "皇姑区黄河南大街70号8门", "另一个句子2", "更多句子..."]
    BBc = BertBase("./bert-base-chinese", "cuda:0")
    BBc.train(sentences1, sentences2, batch_size=32)
