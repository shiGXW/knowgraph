#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
import jieba
import codecs
import gensim.models as word2vec
import numpy as np
import zhconv
from gensim.models.word2vec import LineSentence, logger
from gensim.corpora import WikiCorpus
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy import spatial


def lemmatize(text, tokens, lemmatize, lowercase):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in word_tokenize(text)]


# 从压缩文件中提取出文本
def ExtractText(input_file_name, output_file_name):
    logger.info('主程序开始...')
    logger.info('开始读入wiki数据...')
    input_file = WikiCorpus(input_file_name, tokenizer_func=lemmatize, dictionary={})
    # input_file = WikiCorpus(input_file_name, dictionary={})
    logger.info('wiki数据读入完成！')

    logger.info('处理程序开始...')
    count = 0
    with open(output_file_name, 'wb') as output_file:
        for text in input_file.get_texts():
            output_file.write(' '.join(text).encode("utf-8"))
            output_file.write('\n'.encode("utf-8"))
            count = count + 1
            if count % 10000 == 0:
                logger.info('目前已处理%d条数据' % count)
    logger.info('处理程序结束！')

    output_file.close()
    logger.info('主程序结束！')


# 将繁体中文转化为简体中文
def Trad2Simple(in_file, out_file):

    logger.info('主程序执行开始...')

    input_file = open(in_file, 'r', encoding='utf-8')
    output_file = open(out_file, 'w', encoding='utf-8')

    logger.info('开始读入繁体文件...')
    lines = input_file.readlines()
    logger.info('读入繁体文件结束！')

    logger.info('转换程序执行开始...')
    count = 1
    for line in lines:
        output_file.write(zhconv.convert(line, 'zh-hans'))
        count += 1
        if count % 10000 == 0:
            logger.info('目前已转换%d条数据' % count)
    logger.info('转换程序执行结束！')

    logger.info('主程序执行结束！')


# 分词
def SeparateWords(filePath, fileSegWordDonePath):
    f = codecs.open(filePath, 'r', encoding='utf8')
    target = codecs.open(fileSegWordDonePath, 'w', encoding='utf8')
    logger.info('open files.')

    # 用jieba进行分词
    lineNum = 1
    line = f.readline()
    while line:
        logger.info('---processing ', lineNum, ' article---')
        seg_list = jieba.cut(line, cut_all=False)
        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum + 1
        line = f.readline()
    logger.info("Separate Word Done!!!")
    f.close()
    target.close()


# 训练模型
def train_word2vec(dataset_path, out_model, out_vector):
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    for item in sentences:
        print(item)
    # sentences = LineSentence(smart_open.open(dataset_path, encoding='utf-8'))  # 或者用smart_open打开
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频，sg为1是Skip-Gram模型）
    model = word2vec.Word2Vec(sentences, vector_size=256, sg=1, window=10, min_count=1, workers=3, epochs=50)
    # 保存word2vec模型
    model.save(out_model)
    model.wv.save_word2vec_format(out_vector, binary=False)
    logger.info("Train Done!!!")


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语最相似的词
def calculate_most_similar(model, word):
    similar_words = model.wv.most_similar(word)
    # logger.info(word)
    for term in similar_words:
        return term[0], term[1]


# 计算两个词相似度
def calculate_words_similar(model, word1, word2):
    return model.wv.similarity(word1, word2)


# 找出不合群的词
def find_word_dismatch(model, list):
    return model.wv.doesnt_match(list)


# 计算两个句子相似度
def calculate_sentence_similar1(model, index2word_set, sentence1, sentence2):
    avg_fv1 = avg_feature_vector(sentence1, model, index2word_set, num_features=256)
    avg_fv2 = avg_feature_vector(sentence2, model, index2word_set, num_features=256)
    return 1 - spatial.distance.cosine(avg_fv1, avg_fv2)


def avg_feature_vector(sentence, model, index2word_set, num_features):
    words = jieba.cut(sentence, cut_all=False)
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


# 计算两个句子相似度
def calculate_sentence_similar2(model, index2word_set, sentence1, sentence2):
    # 分词
    vector1 = build_sentence_vector(sentence1, 256, model)
    vector2 = build_sentence_vector(sentence2, 256, model)
    a = np.array(vector1).reshape(-1)
    b = np.array(vector2).reshape(-1)
    return 1 - spatial.distance.cosine(a, b)


# sentence是输入的句子，size是词向量维度，w2v_model是训练好的词向量模型
def build_sentence_vector(sentence, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in jieba.cut(sentence, cut_all=False):
        try:
            vec += w2v_model.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


if __name__ == '__main__':
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='word2vec.log')
    # 从压缩文件中提取出文本
    # ExtractText('./datasets/knowgraph/zhwiki-latest-pages-articles.xml.bz2', './datasets/knowgraph/wiki.zh.txt')
    # # 将繁体中文转化为简体中文
    # Trad2Simple('./datasets/knowgraph/wiki.zh.txt', './datasets/knowgraph/wiki.zh.simple.txt')
    # # 分词
    # SeparateWords('./datasets/knowgraph/wiki.zh.simple.txt', './datasets/knowgraph/wiki.zh.simple.seg.txt')
    # # 分词处理后文本
    # dataset_path = './datasets/knowgraph/wiki.zh.simple.seg.txt'
    # # 词向量
    # out_vector = './datasets/knowgraph/wiki.zh.text.vector'
    # # 模型
    # out_model = './datasets/knowgraph/wiki.zh.text.model'
    # # 训练模型
    # train_word2vec(dataset_path, out_model, out_vector)

    model = load_word2vec_model("./datasets/knowgraph/wiki.zh.text.model")  # 加载模型

    index2word_set = set(model.wv.index_to_key)

    similar1 = calculate_sentence_similar2(model, index2word_set, "玻璃纤维", "玻璃纤维及其制品")

    similar2 = calculate_sentence_similar2(model, index2word_set, "玻璃纤维", "玻璃纤维纱")

    similar3 = calculate_sentence_similar2(model, index2word_set, "玻璃纤维", "玻璃纤维布")

    similar4 = calculate_sentence_similar2(model, index2word_set, "玻璃纤维", "玻璃纤维毡")

    similar5 = calculate_sentence_similar2(model, index2word_set, "玻璃纤维", "玻璃纤维带")

    print(similar1)

    # calculate_most_similar(model, "病毒")  # 找相近词

    # calculate_words_similar(model, "法律", "制度")  # 两个词相似度

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

    # logger.info(model.wv.__getitem__('男人'))  # 词向量

    # list = ["早饭", "吃饭", "恰饭", "嘻哈"]

    # find_word_dismatch(model, list)