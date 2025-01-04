#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import re

import numpy as np
import pandas as pd
from mysqlIn import *


# 读取csv
def read_csv(data_path, indexs):
    excel_datas = []
    for index in indexs:
        file_name = data_path + str(index) + ".csv"
        try:
            excel_data = pd.read_csv(file_name, header=0, encoding="gbk")
        except:
            excel_data = pd.read_csv(file_name, header=0, encoding="utf-8")
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


# 划分数据集
def dataset_part(excel_datas):
    # excel_datas[0][0] = {
    #     "enterprise": [],
    #     "industry": [],
    # }
    # excel_datas[0][1...] = {
    #     "enterprise": [],
    #     "Ematerial": [],
    #     "Eproduct": [],
    #     "Ekeyword": [],
    #     "Etechnology": [],
    # }
    industry_dict = {item: [] for item in excel_datas[0][0][list(excel_datas[0][0].keys())[1]]}
    industry_train_dict = {item: [] for item in excel_datas[0][0][list(excel_datas[0][0].keys())[1]]}
    industry_valid_dict = {item: [] for item in excel_datas[0][0][list(excel_datas[0][0].keys())[1]]}
    # 字典数据写入数据集字典
    for item in range(excel_datas[0][1]):
        industry_dict[excel_datas[0][0][list(excel_datas[0][0].keys())[1]][item]].append(item)
    # 划分
    for key in list(industry_dict.keys()):
        if len(industry_dict[key]) > 1:
            result = split_list(industry_dict[key], train_ratio, 2)
        industry_train_dict[key] = result[0]
        industry_valid_dict[key] = result[1]
    # 数据集列表
    # excel_datas[0][0] 中 enterprise 的元素位置
    industry_train_list = [x for sublist in list(industry_train_dict.values()) for x in sublist]
    industry_valid_list = [x for sublist in list(industry_valid_dict.values()) for x in sublist]
    return industry_train_list, industry_valid_list


def split_list(lst, ratios, num_splits):

    """
    将列表按照指定比例和数量拆分成子列表
    :param lst: 待拆分列表
    :param ratios: 每个子列表的元素占比，由小数表示的列表
    :param num_splits: 子列表的数量
    :return: 拆分后的子列表组成的列表
    """
    if len(ratios) != num_splits:
        raise ValueError("The length of ratios must equal to num_splits.")
    total_ratio = sum(ratios)
    if total_ratio != 1:
        raise ValueError("The sum of ratios must be equal to 1.")
    n = len(lst)
    result = []
    start = 0
    for i in range(num_splits):
        end = start + int(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result


def dealRawAcc(excel_data):
    enters_ex = []
    rawaccs_ex = []
    data_except = []
    delimiters = ["、", "（", "）"]
    for item in range(len(excel_data[list(excel_data.keys())[0]])):
        try:
            # 分割内容
            regex_pattern = '|'.join(map(re.escape, delimiters))
            rawacc_list = re.split(regex_pattern, excel_data[list(excel_data.keys())[1]][item].strip().replace(" ", ""))
            for rawacc in rawacc_list:
                if rawacc != "":
                    enters_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    rawaccs_ex.append(rawacc)
        except:
            data_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    return enters_ex, rawaccs_ex


def write_txt(excel_datas, data_list, flag, indexs):
    data_triples = {
        "sub": [],
        "rel": [],
        "obj": [],
    }
    rel_list = ["industry", "wasteHW", "waste", "material", "product"]
    for index in indexs:
        # [0, 1, 2]，list 为读取到的数据在 excel_datas 中的位置
        if index in [0]:
            for item in data_list:
                data_triples["sub"].append(excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item])
                data_triples["rel"].append(rel_list[index])
                data_triples["obj"].append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item])
        else:
            planid_data = dealRawAcc(excel_datas[index][0])
            plan_deal_sub_list, plan_deal_obj_list = plan_id_deal(excel_datas, planid_data, 3, data_list)
            for item in range(len(plan_deal_sub_list)):
                data_triples["sub"].append(plan_deal_sub_list[item])
                data_triples["rel"].append(rel_list[index])
                data_triples["obj"].append(plan_deal_obj_list[item])

    # 导出到文本文件
    pd.DataFrame({0: data_triples["sub"], 1: data_triples["rel"], 2: data_triples["obj"]}).to_csv(
        f'./data/{flag}.txt', sep='\t', index=False
    )


def plan_id_deal(excel_datas, planid_data, index, data_list):
    sub_list = []
    obj_list = []
    for item in range(len(planid_data[0])):
        for planid in excel_datas[index][0]["planId"]:
            if planid == planid_data[0][item]:
                sub_list.append(excel_datas[index][0]["enterpriseId"])
                sub_list.append(planid_data[1][item])

    return sub_list, obj_list


if __name__ == '__main__':
    train_ratio = [0.8, 0.2]
    # 获取数据
    excel_path = r"./data/gcndata/"
    # list 为文件名，0,1,2,3,4,5,6,
    excel_datas = read_csv(excel_path, [0, 1, 2, 3, 4, 5, 6])
    # id_enterprise 对应信息字典写入 json
    id_enterprise_dict = dict(zip(excel_datas[6][0]["entid"], excel_datas[6][0]["entName"]))
    id_enterprise_dict_json_str = json.dumps(id_enterprise_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'id_enterprise_dict.json'), 'w') as json_file:
        json_file.write(id_enterprise_dict_json_str)
    # 划分数据集
    industry_train_list, industry_valid_list = dataset_part(excel_datas)
    # 训练数据写入 txt，list 为读取到的数据在 excel_datas 中的位置
    write_txt(excel_datas, industry_train_list, "train", [0, 1, 2, 4, 5])
    write_txt(excel_datas, industry_valid_list, "valid", [0, 1, 2, 4, 5])
    # print(dataenterdict)
