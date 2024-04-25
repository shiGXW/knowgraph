#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import logging
import numpy as np
from transformers import BertTokenizer, BertModel


class BertBaseChinese():
    def __init__(self, model_name, device):
        # 设置输出日志
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                            filename='BERT.log')

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

    def train(self, sentences1, sentences2, batch_size):
        # 按照每批32个句子对计算相似度
        similarities = self.batch_compare_sentences(sentences1, sentences2, batch_size=32)

        return similarities

        # # 输出信息
        # for i in range(0, len(sentences1), batch_size):
        #     current_batch_similarities = similarities[i:i + batch_size]
        #
        #     # 输出当前批次的每个句子对及其相似度
        #     for j in range(min(batch_size, len(sentences1) - i)):
        #         logging.info(f"句1: {sentences1[i + j]}")
        #         logging.info(f"句2: {sentences2[i + j]}")
        #         logging.info(f"相似度: {current_batch_similarities[j]:.4f}\n")  # 控制相似度小数点后四位精度
        #
        #     # 在批次之间空一行
        #     if i + batch_size < len(sentences1):
        #         logging.info("\n")
        #
        # # 若最后一个批次不满batch_size，也正常处理
        # remaining = len(sentences1) - (len(sentences1) // batch_size * batch_size)
        # if remaining > 0:
        #     last_batch_similarities = similarities[-remaining:]
        #     for j in range(remaining):
        #         logging.info(j)
        #         logging.info(f"句1: {sentences1[-remaining + j]}")
        #         logging.info(f"句2: {sentences2[-remaining + j]}")
        #         logging.info(f"相似度: {last_batch_similarities[j]:.4f}\n")

    def batch_compare_sentences(self, sentences1, sentences2, batch_size=32):
        # 确保 sentences1 和 sentences2 的长度相等
        assert len(sentences1) == len(sentences2)

        # 初始化存储结果的容器
        similarities = []

        # 将句子按照批次拆分
        for i in range(0, len(sentences1), batch_size):
            batch1 = sentences1[i:i + batch_size]
            batch2 = sentences2[i:i + batch_size]

            # 使用tokenizer对批次数据进行处理
            encoded_inputs1 = self.tokenizer(
                batch1,  # 即将被编码的批次句子
                padding=True,  # 自动填充句子为最大长度，默认用 0 填充
                truncation=True,  # 超过max_length指定的限制时截断
                max_length=self.model.config.max_position_embeddings,
                return_tensors="pt"
            )
            encoded_inputs2 = self.tokenizer(
                batch2,
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
                return_tensors="pt"
            )

            input_ids1 = encoded_inputs1['input_ids'].to(self.device)
            attention_mask1 = encoded_inputs1['attention_mask'].to(self.device)
            input_ids2 = encoded_inputs2['input_ids'].to(self.device)
            attention_mask2 = encoded_inputs2['attention_mask'].to(self.device)

            # 获取词向量，在GPU上运行模型
            with torch.no_grad():
                outputs1 = self.model(input_ids1, attention_mask=attention_mask1)
                outputs2 = self.model(input_ids2, attention_mask=attention_mask2)

            # 提取句子向量
            sentence_embeddings1 = outputs1[0][:, 0, :]
            sentence_embeddings2 = outputs2[0][:, 0, :]

            # 计算余弦相似度
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
            batch_similarities = cos(sentence_embeddings1, sentence_embeddings2)

            # 将批次内每个句子对的相似度添加到结果容器
            similarities.extend(batch_similarities.cpu().numpy())

        # 返回所有批次计算出的相似度结果
        return np.array(similarities)


if __name__ == '__main__':
    # 使用示例
    # 假设有足够多的句子
    sentences1 = ["玻璃纤维", "玻璃纤维", "玻璃纤维", "玻璃纤维", "玻璃纤维", "黄河南大街70号8门", "另一个句子1", "更多句子..."]
    # 与sentences1长度相同
    sentences2 = ["玻璃纤维及其制品", "玻璃纤维纱", "玻璃纤维布", "玻璃纤维毡", "玻璃纤维带", "皇姑区黄河南大街70号8门", "另一个句子2", "更多句子..."]
    BBc = BertBaseChinese("./bert-base-chinese", "cuda:0")
    BBc.train(sentences1, sentences2, batch_size=32)
