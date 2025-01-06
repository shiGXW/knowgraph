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
from BERTDDD import *
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
    catalogue_match_datas = []
    for index in indexs:
        if index in ["5"]:
            file_name = data_path + str(index) + ".txt"
            try:
                data = pd.read_csv(file_name, delimiter='\t', encoding="gbk", dtype=str)
            except:
                data = pd.read_csv(file_name, delimiter='\t', encoding="utf-16", dtype=str)
        else:
            file_name = data_path + str(index) + ".xlsx"
            try:
                data = pd.read_excel(file_name, header=None, dtype=str)
            except:
                data = pd.read_excel(file_name, header=None, dtype=str)
            data = pd.DataFrame(
                data[[0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 49, 50, 51]].to_numpy()[0:]
            )
        # 数据字典
        data_dict = {item: [] for item in data.columns}
        # 数据写入字典
        for item in data.index:
            for column in data.columns:
                data_dict[column].append(data[column][item])
        catalogue_match_datas.append((data_dict, data.shape[0]))
    return catalogue_match_datas

# 原料、产品实体链接
def RawAcc_BEONE(excel_datas_merge, catalogue_match_datas):
    # rawacc_beone_dict = {}
    # id_enterprise 对应信息字典写入 json
    rawacc_beone_dict = {"saved_current": -1}
    data_except = []
    delimiters = ["，", "、", "（", "）"]
    excel_data_original_all = excel_datas_merge[6][0][1] + excel_datas_merge[7][0][1]
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
    catalogue_data = catalogue_match_datas[0][0][str(5)]
    match_data = [str(item) for sublist in list(zip(*list(catalogue_match_datas[1][0].values()))) for item in sublist]
    match_data_simple = [item for item in match_data if item != "nan" and item != "no"]
    del excel_datas_merge, catalogue_match_datas
    logging.info(f"excel_data：{len(excel_data)}")
    logging.info(f"catalogue_data：{len(catalogue_data)}")
    logging.info(f"match_data：{len(match_data)}")
    logging.info(f"match_data_simple：{len(match_data_simple)}")

    # 加载模型
    BBc = BertBase("../../datasets/knowgraph/bert-base-chinese/", "cuda:1", catalogue_data, match_data_simple, batch_size=1024)

    # 进度条及预估时间
    start_time = time.time()
    current = -1
    begin = 0
    total = len(excel_data)
    # total = 1 + len(excel_data)//100

    # 是否继续之前保存
    if os.path.exists(os.path.join(excel_path, 'rawacc_beone_max_dict_simple.json')):
        with open(os.path.join(excel_path, 'rawacc_beone_max_dict_simple.json'), 'r') as json_file:
            rawacc_beone_dict = json.loads(str(json_file.read()))
        begin = rawacc_beone_dict["saved_current"]

    logging.info(f"Begin index: {begin}")

    # for excel_data_item in [excel_data[i:i+1] for i in range(0, len(excel_data), 1)]:
    for current in range(begin, total):

        excel_data_item = [excel_data[current]]
        progress_bar(start_time, total, current)

        # 输出及保存
        if current % 10 == 0 and current != rawacc_beone_dict["saved_current"] and current != 0:
            # 中途保存
            rawacc_beone_dict["saved_current"] = current
            rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
            with open(os.path.join(excel_path, 'rawacc_beone_max_dict_simple.json'), 'w') as json_file:
                json_file.write(rawacc_beone_dict_json_str)
            logging.info(f"saved {current}")

        similarities_catalogue, similarity_catalogue_indexs = BBc.train(excel_data_item, "catalogue", batch_size=1024)
        similarities_match, similarity_match_indexs = BBc.train(excel_data_item, "match", batch_size=1024)
        for index, similar in enumerate(similarities_catalogue):
            # 构建匹配映射，首先匹配catalogue_datas；再匹配手工匹配度；比较相似度，选最优
            # 匹配catalogue_data相似度高
            if similarities_catalogue[index] >= similarities_match[index]:
                if similarities_catalogue[index] >= 0.9:
                    rawacc_beone_dict[excel_data_item[index]] = catalogue_data[similarity_catalogue_indexs[index]]
                else:
                    data_except.append(excel_data_item[index])
            # 匹配match_data_simple相似度高
            else:
                if similarities_match[index] >= 0.9:
                    rawacc_beone_dict[excel_data_item[index]] = match_data[
                        22*(match_data.index(match_data_simple[similarity_match_indexs[index]])//22)
                    ]
                else:
                    data_except.append(excel_data_item[index])
    rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'rawacc_beone_max_dict_simple.json'), 'w') as json_file:
        json_file.write(rawacc_beone_dict_json_str)


# 进度条函数
def progress_bar(start_time, total, current):
    bar_length = 50  # 进度条的长度
    percent = floor(current / total * 100)
    if percent > 100:
        percent = 100
    bar = ''.join(["#" for _ in range(int(bar_length * current / total))])
    logging.info("\r[{}->{}] {}% {}".format(bar, ' ' * (bar_length - len(bar)), percent, time_to_complete(start_time, total, current)))


# 估算剩余时间
def time_to_complete(start_time, total, current):
    if current > 0:
        dur_per_item = (time.time() - start_time) / current
        remaining = total - current
        m, s = divmod(dur_per_item * remaining, 60)
        h, m = divmod(m, 60)
        return "Remaining: %02d:%02d:%02d" % (h, m, s)
    else:
        return "Estimated time to complete"


if __name__ == '__main__':
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 获取数据
    excel_path = r"../../datasets/knowgraph/maxDDD/"
    catalogue_match_datas = read_csv(excel_path, ["5", "17"])
    # 读取全部数据
    with open(os.path.join(excel_path, 'excel_datas_merge.txt'), 'r') as file:
        excel_datas_merge = eval(file.read())

    # 原料、产品实体链接：rawacc_beone_max_dict_simple.json
    RawAcc_BEONE(excel_datas_merge, catalogue_match_datas)

    logging.info("Done!!!")
