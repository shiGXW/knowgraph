#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import os
import re
import json
import time
import jieba
from jieba import analyse
import numpy as np
from math import nan
import pandas as pd
from BERTDD import *
from math import floor
from ordered_set import OrderedSet
import urllib.request
import urllib.parse
import json
import requests
import random
import hashlib
import re

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'


# 读取csv
def read_csv(data_path, indexs):
    excel_datas = []
    for index in indexs:
        if index in ["5"]:
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
        # 数据字典
        data_dict = {item: [] for item in excel_data.columns}
        # 数据写入字典
        for item in excel_data.index:
            for column in excel_data.columns:
                data_dict[column].append(excel_data[column][item])
        excel_datas.append((data_dict, excel_data.shape[0]))
    return excel_datas


# 原料、产品实体链接
def RawAcc_BEONE(excel_datas, catalogue_datas):
    rawacc_beone_dict = {}
    data_except = []
    delimiters = ["、", "（", "）"]
    # 加载模型
    BBc = BertBase("../datasets/knowgraph/bert-base-chinese/", "cuda:1")
    excel_data_original_all = excel_datas[6][0][1] + excel_datas[6][0][1]
    logging.info(f"excel_data_all：{len(excel_data_original_all)}")
    # 加载数据，去重
    excel_data_original = list(set(excel_data_original_all))
    # 数据清洗
    excel_data = []
    for item in excel_data_original:
        # 分割内容
        regex_pattern = '|'.join(map(re.escape, delimiters))
        rawacc_list = re.split(regex_pattern, item.strip().replace(" ", ""))
        excel_data = excel_data + rawacc_list
    catalogue_data = catalogue_datas[0][0][str(5)]
    del excel_datas, catalogue_datas
    logging.info(f"excel_data：{len(excel_data)}")
    logging.info(f"catalogue_data：{len(catalogue_data)}")
    similarities, similarity_indexs = BBc.train(excel_data, catalogue_data, batch_size=1024)
    for index, similar in enumerate(similarities):
        if similarities[index] >= 0.9:
            rawacc_beone_dict[excel_data[index]] = catalogue_data[similarity_indexs[index]]
        else:
            data_except.append(excel_data[index])
    rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'rawacc_beone_max_dict_simple.json'), 'w') as json_file:
        json_file.write(rawacc_beone_dict_json_str)


if __name__ == '__main__':
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 获取数据
    excel_path = r"../datasets/knowgraph/max/"
    catalogue_datas = read_csv(excel_path, ["5"])
    # 读取全部数据
    with open(os.path.join(excel_path, 'excel_datas.txt'), 'r') as file:
        excel_datas = eval(file.read())

    # 原料、产品实体链接：rawacc_beone_max_dict_simple.json
    RawAcc_BEONE(excel_datas, catalogue_datas)

    logging.info("Done!!!")
