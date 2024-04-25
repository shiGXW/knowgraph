#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import logging
import re
import sys
import pandas as pd
import gensim.models as word2vec
from gensim.models.word2vec import LineSentence, logger


# 读取csv
def read_csv(data_path, indexs):
    excel_datas = []
    for index in indexs:
        if index in [7, 8]:
            file_name = data_path + str(index) + ".txt"
            try:
                excel_data = pd.read_csv(file_name, delimiter='\t', encoding="gbk", dtype=str)
            except:
                excel_data = pd.read_csv(file_name, delimiter='\t', encoding="utf-16", dtype=str)
        else:
            file_name = data_path + str(index) + ".csv"
            try:
                excel_data = pd.read_csv(file_name, header=0, encoding="gbk", dtype=str)
            except:
                excel_data = pd.read_csv(file_name, header=0, encoding="utf-8", dtype=str)
        # dataenterdict = {
        #     "enterprise": [],
        # }
        # 数据字典
        data_dict = {item: [] for item in excel_data.columns}
        # 数据写入字典
        for item in excel_data.index:
            for column in excel_data.columns:
                data_dict[column].append(excel_data[column][item])
        excel_datas.append((data_dict, excel_data.shape[0]))
    return excel_datas


# 分词
def SeparateWords(excel_datas, fileSegWordDonePath):
    logger.info('open files.')
    data_except = []
    delimiters = ["、", "（", "）"]
    rawacc_all = []
    for index in [1, 2]:
        for item in range(len(excel_datas[index][0][list(excel_datas[index][0].keys())[1]])):
            try:
                # 分割内容
                regex_pattern = '|'.join(map(re.escape, delimiters))
                rawacc_list = re.split(regex_pattern, excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item].strip().replace(" ", ""))
                for rawacc in rawacc_list:
                    if rawacc:
                        rawacc_all.append(rawacc)
                    else:
                        data_except.append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item])
            except:
                data_except.append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item])
                continue

    target = codecs.open(fileSegWordDonePath, 'w', encoding='utf8')

    target_data = ' '.join(rawacc_all + excel_datas[0][0][str(5)])
    target.writelines(target_data)
    logger.info("Separate Word Done!!!")
    target.close()


# 训练模型
def train_word2vec(dataset_path, out_model, out_vector):
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # sentences = LineSentence(smart_open.open(dataset_path, encoding='utf-8'))  # 或者用smart_open打开
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频，sg为1是Skip-Gram模型）
    model = word2vec.Word2Vec([sentences], vector_size=256, sg=1, window=10, min_count=5, workers=3, epochs=50)
    # 保存word2vec模型
    model.save(out_model)
    model.wv.save_word2vec_format(out_vector, binary=False)
    logger.info("Train Done!!!")


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语最相似的词
def calculate_most_similar(self, word):
    similar_words = self.wv.most_similar(word)
    # logger.info(word)
    for term in similar_words:
        return term[0], term[1]


# 计算两个词相似度
def calculate_words_similar(self, word1, word2):
    return self.wv.similarity(word1, word2)


# 找出不合群的词
def find_word_dismatch(self, list):
    return self.wv.doesnt_match(list)


if __name__ == '__main__':
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='word2vec.log')
    # # 获取文本
    # excel_path = r"./datasets/knowgraph/"
    # excel_datas = read_csv(excel_path, [8, 4, 5])
    # # 分词
    # SeparateWords(excel_datas, './datasets/knowgraph/RawAcc.seg.txt')
    # 分词处理后文本
    dataset_path = './datasets/knowgraph/RawAcc.seg.txt'
    # 词向量
    out_vector = './datasets/knowgraph/RawAcc.text.vector'
    # 模型
    out_model = './datasets/knowgraph/RawAcc.text.model'
    # 训练模型
    train_word2vec(dataset_path, out_model, out_vector)

    # model = load_word2vec_model("./datasets/knowgraph/RawAcc.text.model")  # 加载模型
    #
    # similar1 = calculate_words_similar(model, "玻璃纤维", "玻璃纤维及其制品")
    #
    # similar2 = calculate_words_similar(model, "玻璃纤维", "玻璃纤维纱")
    #
    # similar3 = calculate_words_similar(model, "玻璃纤维", "玻璃纤维布")
    #
    # similar4 = calculate_words_similar(model, "玻璃纤维", "玻璃纤维毡")
    #
    # similar5 = calculate_words_similar(model, "玻璃纤维", "玻璃纤维带")
    #
    # print(similar1)

    # calculate_most_similar(model, "病毒")  # 找相近词

    # calculate_words_similar(model, "法律", "制度")  # 两个词相似度

    # logger.info(model.wv.__getitem__('男人'))  # 词向量

    # list = ["早饭", "吃饭", "恰饭", "嘻哈"]

    # find_word_dismatch(model, list)