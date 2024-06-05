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
from BERT import *
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
    # id_enterprise 对应信息字典写入 json
    rawacc_beone_dict = {"saved_current": -1}
    data_item_except = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 加载模型
    BBc = BertBase("../datasets/knowgraph/bert-base-chinese/", "cuda:1")
    excel_data_all = excel_datas[6][0][1] + excel_datas[6][0][1]
    logging.info(f"excel_data_all：{len(excel_data_all)}")
    # 加载数据，去重
    excel_data = list(set(excel_data_all))
    del excel_datas
    catalogue_data = catalogue_datas[0][0][str(5)]
    logging.info(f"excel_data：{len(excel_data)}")
    logging.info(f"catalogue_data：{len(catalogue_data)}")
    # 进度条及预估时间
    start_time = time.time()
    current = -1
    begin = 0
    total = len(excel_data)
    # 是否继续之前保存
    if os.path.exists(os.path.join(excel_path, 'rawacc_beone_max_dict.json')):
        with open(os.path.join(excel_path, 'rawacc_beone_max_dict.json'), 'r') as json_file:
            rawacc_beone_dict = json.loads(str(json_file.read()))
        begin = rawacc_beone_dict["saved_current"]
    logging.info(f"Begin index: {begin}")
    for current in range(begin, total):
        # 输出及保存
        if current % 10 == 0 and current != rawacc_beone_dict["saved_current"] and current != 0 or current == total-1:
            progress_bar(start_time, total, current)
            # 中途保存
            rawacc_beone_dict["saved_current"] = current
            rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
            with open(os.path.join(excel_path, 'rawacc_beone_max_dict.json'), 'w') as json_file:
                json_file.write(rawacc_beone_dict_json_str)
            logging.info(f"saved {current}")

        # 分割内容
        regex_pattern = '|'.join(map(re.escape, delimiters))
        rawacc_list = re.split(regex_pattern, excel_data[current].strip().replace(" ", ""))
        for rawacc in rawacc_list:
            if rawacc and rawacc not in rawacc_beone_dict:
                rawacc_temp = None
                match_ratio = 0.0

                similarities = BBc.train(
                    [rawacc for _ in range(len(catalogue_data))],
                    catalogue_data, batch_size=1024 * 13
                )

                for index, similar in enumerate(similarities):
                    if similar > match_ratio:
                        match_ratio = similar
                        rawacc_temp = catalogue_data[index]

                if match_ratio >= 0.9:
                    rawacc_beone_dict[rawacc] = rawacc_temp
                else:
                    data_except.append(rawacc)
            else:
                data_item_except.append(excel_data[current])


# 原料、产品实体链接
def RawAcc_BEONE_E(excel_datas):
    # id_enterprise 对应信息字典写入 json
    rawacc_beone_dict = {"saved_current": -1}
    data_item_except = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 加载模型
    BBc = BertBase("../datasets/knowgraph/bert-base-uncased/", "cuda:1")
    # 加载数据，去重
    excel_data = list(set(excel_datas[6][0][1] + excel_datas[6][0][1]))
    del excel_datas
    catalogue_data = catalogue_datas[0][0][str(5)]
    with open(os.path.join(excel_path, 'catalogue_dataE_dict.json'), 'r') as json_file:
        catalogue_dataE = json.loads(str(json_file.read()))
    with open(os.path.join(excel_path, 'catalogue_dataER_dict.json'), 'r') as json_file:
        catalogue_dataER = json.loads(str(json_file.read()))
    logging.info(f"excel_data_data：{len(excel_data)}")
    logging.info(f"catalogue_data：{len(catalogue_data)}")
    # 进度条及预估时间
    start_time = time.time()
    current = -1
    begin = 0
    total = len(excel_data)
    # 是否继续之前保存
    if os.path.exists(os.path.join(excel_path, 'rawacc_beone_E_max_dict.json')):
        with open(os.path.join(excel_path, 'rawacc_beone_E_max_dict.json'), 'r') as json_file:
            rawacc_beone_dict = json.loads(str(json_file.read()))
        begin = rawacc_beone_dict["saved_current"]
    logging.info(f"Begin index: {begin}")
    for current in range(begin, total):
        # 输出及保存
        if current % 10 == 0 and current != rawacc_beone_dict["saved_current"] and current != 0:
            progress_bar(start_time, total, current)
            # 中途保存
            rawacc_beone_dict["saved_current"] = current
            rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
            with open(os.path.join(excel_path, 'rawacc_beone_E_max_dict.json'), 'w') as json_file:
                json_file.write(rawacc_beone_dict_json_str)
            logging.info(f"saved {current}")

        # 分割内容
        regex_pattern = '|'.join(map(re.escape, delimiters))
        rawacc_list = re.split(regex_pattern, excel_data[current].strip().replace(" ", ""))
        for rawacc in rawacc_list:
            if rawacc and rawacc not in rawacc_beone_dict:
                rawaccE = baidu_translate(rawacc)
                rawacc_temp = None
                match_ratio = 0.0

                similarities = BBc.train(
                    [rawaccE for _ in range(len(catalogue_data))],
                    catalogue_dataE, batch_size=1024 * 13
                )

                for index, similar in enumerate(similarities):
                    if similar > match_ratio:
                        match_ratio = similar
                        rawacc_temp = catalogue_dataER[index]

                if match_ratio >= 0.9:
                    rawacc_beone_dict[rawacc] = rawacc_temp
                else:
                    data_except.append(rawacc)
            else:
                data_item_except.append(excel_data[current])


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


# 百度翻译方法
def baidu_translate(content):
    print(content)
    if len(content) > 4891:
        return '输入请不要超过4891个字符！'
    salt = str(random.randint(0, 50))
    # 申请网站 http://api.fanyi.baidu.com/api/trans
    appid = '20221028001421570'
    secretKey = 'j5sOtXVXJSMwWPuH5azm'
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    head = {'q': f'{content}',
            'from': 'zh',
            'to': 'en',
            'appid': f'{appid}',
            'salt': f'{salt}',
            'sign': f'{sign}'}
    j = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', head)
    # print(j.json())
    res = j.json()['trans_result'][0]['dst']
    res = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', res)
    # print(res)
    return res


def catalogue_data_tran(excel_datas):
    catalogue_data = excel_datas[4][0][str(5)]
    catalogue_dataE = []
    for item in catalogue_data:
        time.sleep(1)
        catalogue_dataE.append(baidu_translate(item))
    # catalogue_data 与 catalogue_dataE 对应信息字典写入 json
    catalogue_dataE_dict = dict(zip(catalogue_data, catalogue_dataE))
    catalogue_dataER_dict = dict(zip(catalogue_dataE, catalogue_data))
    catalogue_dataE_dict_json_str = json.dumps(catalogue_dataE_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'catalogue_dataE_dict.json'), 'w') as json_file:
        json_file.write(catalogue_dataE_dict_json_str)
    catalogue_dataER_dict_json_str = json.dumps(catalogue_dataER_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'catalogue_dataER_dict.json'), 'w') as json_file:
        json_file.write(catalogue_dataER_dict_json_str)


if __name__ == '__main__':
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # # 获取数据
    # excel_path = r"../datasets/knowgraph/"
    # # list 为文件名，0,1,2,3,4,5,6,
    # excel_datas = read_csv(excel_path, ["2021/4", "2021/5", "2022/3", "2022/4", "2022/8"])

    # 获取数据
    excel_path = r"../datasets/knowgraph/max/"
    catalogue_datas = read_csv(excel_path, ["5"])
    # 读取全部数据
    # "material", "product"：excel_datas[6]、excel_datas[7]
    with open(os.path.join(excel_path, 'excel_datas.txt'), 'r') as file:
        excel_datas = eval(file.read())

    # 原料、产品实体链接：rawacc_beone_max_dict.json
    RawAcc_BEONE(excel_datas, catalogue_datas)

    # # 翻译
    # catalogue_data_tran(excel_datas)

    # # 原料、产品实体链接：rawacc_beone_E_max_dict.json
    # RawAcc_BEONE_E(excel_datas)

    logging.info("Done!!!")
