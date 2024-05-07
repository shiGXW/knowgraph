#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import nltk
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument, logger
from nltk.tokenize import word_tokenize
import logging


# -------------Let’s prepare data for training our doc2vec model----------------------
# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='Doc2vec.log')
# 分词处理后文本
dataset_path = './datasets/knowgraph/wiki.zh.simple.txt'
# 词向量
out_vector = './datasets/knowgraph/wiki.zh.text.Doc.vector'
# 模型
out_model = './datasets/knowgraph/wiki.zh.text.Doc.model'
# -------------Lets start training our model----------------------

logger.info("running %s" % ' '.join(sys.argv))

# 设置迭代器
tagged_data = TaggedLineDocument(dataset_path)

max_epochs = 50
vec_size = 256
alpha = 0.025
# dm=1 :保留了文档中的单词顺序 dm=0 :不保留任何语序
model = Doc2Vec(dm=1, vector_size=vec_size, window=10, alpha=alpha, min_count=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save(out_model)
print("Model Saved")


# -------------Lets play with it----------------------
model = Doc2Vec.load(out_model)
# to find the vector of a document which is not in training data
test_data = word_tokenize("玻璃纤维及其制品")
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
