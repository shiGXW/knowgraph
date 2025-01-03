#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import os
import re
import json
import time
import jieba
import numpy as np
import pandas as pd
import logging
from math import nan
from math import floor
from ordered_set import OrderedSet
from collections import defaultdict as ddict


# 读取csv
def read_csv(data_path, indexs):
    excel_datas_merge = []
    for index in indexs:
        if index in ["4", "5", "industry", "areacode", "HW", "waste", "material", "product", "HWwaste"]:
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
        excel_datas_merge.append((data_dict, excel_data.shape[0]))
    logging.info("read excels")
    return excel_datas_merge


def data_merge(excel_datas_merge_max):
    excel_datas_merge = []
    for iindex, item in enumerate(["industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"]):
        # 数据字典
        data_dict = {item: [] for item in [0, 1]}
        if iindex in [0, 1, 2, 3]:
            logging.info("baseinfo")
            for index, data in enumerate(excel_datas_merge_max[iindex][0][list(excel_datas_merge_max[iindex][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[iindex][0][list(excel_datas_merge_max[iindex][0].keys())[1]][index])
        elif item == "HW":
            logging.info("HW")
            for index, data in enumerate(excel_datas_merge_max[11][0][list(excel_datas_merge_max[11][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[11][0][list(excel_datas_merge_max[11][0].keys())[1]][index])
            for index, data in enumerate(excel_datas_merge_max[6][0][list(excel_datas_merge_max[6][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[6][0][list(excel_datas_merge_max[6][0].keys())[1]][index])
        elif item == "waste":
            logging.info("waste")
            for index, data in enumerate(excel_datas_merge_max[12][0][list(excel_datas_merge_max[12][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[12][0][list(excel_datas_merge_max[12][0].keys())[1]][index])
            for index, data in enumerate(excel_datas_merge_max[7][0][list(excel_datas_merge_max[7][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[7][0][list(excel_datas_merge_max[7][0].keys())[1]][index])
        elif item == "material":
            logging.info("material")
            for index, data in enumerate(excel_datas_merge_max[13][0][list(excel_datas_merge_max[13][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[13][0][list(excel_datas_merge_max[13][0].keys())[1]][index])
            # planid 转 entid
            plan_deal_sub_list, plan_deal_obj_list = plan_id_deal(excel_datas_merge_max,  excel_datas_merge_max[9][0])
            for item in range(len(plan_deal_sub_list)):
                data_dict[0].append(plan_deal_sub_list[item])
                data_dict[1].append(plan_deal_obj_list[item])
        elif item == "product":
            logging.info("product")
            for index, data in enumerate(excel_datas_merge_max[14][0][list(excel_datas_merge_max[14][0].keys())[0]]):
                data_dict[0].append(data)
                data_dict[1].append(excel_datas_merge_max[14][0][list(excel_datas_merge_max[14][0].keys())[1]][index])
            # planid 转 entid
            plan_deal_sub_list, plan_deal_obj_list = plan_id_deal(excel_datas_merge_max, excel_datas_merge_max[10][0])
            for item in range(len(plan_deal_sub_list)):
                data_dict[0].append(plan_deal_sub_list[item])
                data_dict[1].append(plan_deal_obj_list[item])
        elif item == "HWwaste":
            logging.info("HWwaste")
            for index, data in enumerate(excel_datas_merge_max[15][0][list(excel_datas_merge_max[15][0].keys())[0]]):
                data_dict[0].append(excel_datas_merge_max[15][0][list(excel_datas_merge_max[15][0].keys())[1]][index])
                data_dict[1].append(data)
            for index, data in enumerate(excel_datas_merge_max[16][0][list(excel_datas_merge_max[16][0].keys())[0]]):
                data_dict[0].append(data[:4])
                data_dict[1].append(excel_datas_merge_max[16][0][list(excel_datas_merge_max[16][0].keys())[1]][index])
        excel_datas_merge.append((data_dict, len(data_dict[0])))

    with open(os.path.join(excel_path, 'excel_datas_merge.txt'), 'w') as file:
        file.write(str(excel_datas_merge))


# 计划id对应企业id，得到三元组头尾
def plan_id_deal(excel_2021_datas, planid_data):
    sub_list = []
    obj_list = []
    data_except = []
    # 遍历数据
    for item in range(len(planid_data[list(planid_data.keys())[0]])):
        # 查询 planid 对应 enterpriseId
        for planid in excel_2021_datas[8][0]["planId"]:
            if planid == planid_data[list(planid_data.keys())[0]][item]:
                # planid 对应的 enterpriseId
                # logging.info(excel_datas_merge[index][0]["planId"].index(planid))
                # logging.info(planid)
                enterpriseId = excel_2021_datas[8][0]["enterpriseId"][excel_2021_datas[8][0]["planId"].index(planid)]
                # logging.info(enterpriseId)
                sub_list.append(enterpriseId)
                obj_list.append(planid_data[list(planid_data.keys())[1]][item])
    logging.info("plan_id_deal")
    logging.info(len(data_except))
    logging.info(f"{len(sub_list)}, {len(obj_list)}")
    return sub_list, obj_list


# 数据统计——企业去重，id对应企业名，id对应md5值
def data_stat_id_md5(excel_datas_merge):
    id_md5_dict = {}
    for index, enter in enumerate(excel_datas_merge[1][0][1]):
        # id 加密，企业的 md5 值
        md5_hash = hashlib.md5()
        md5_hash.update(enter.encode("utf-8"))
        id_md5_dict[excel_datas_merge[1][0][0][index]] = md5_hash.hexdigest()
    # id_enterprise 对应信息字典写入 json
    id_enterprise_dict = dict(zip(excel_datas_merge[1][0][0], excel_datas_merge[1][0][1]))
    id_enterprise_dict_json_str = json.dumps(id_enterprise_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'id_enterprise_dict.json'), 'w') as json_file:
        json_file.write(id_enterprise_dict_json_str)

    # id_md5 对应信息字典写入 json
    id_md5_dict_json_str = json.dumps(id_md5_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'id_md5_dict.json'), 'w') as json_file:
        json_file.write(id_md5_dict_json_str)


# 数据统计——行业统计
def data_stat_industry(excel_datas_merge, class_industrys):
    # 行业大类
    industry_class_B = {}
    # 行业小类
    industry_class_S = {}
    industry_class_except = []
    # 获取行业频次
    for industry in excel_datas_merge[0][0][1]:
        try:
            inencode = get_indust_encode(class_industrys, industry)
            if industry in industry_class_S and inencode[0:2] in industry_class_B:
                industry_class_S[industry] = industry_class_S[industry] + 1
                industry_class_B[inencode[0:2]] = industry_class_B[inencode[0:2]] + 1
            else:
                industry_class_S[industry] = 1
                industry_class_B[inencode[0:2]] = 1
        except:
            industry_class_except.append(industry)
            continue
    industry_class_B_sort_dict = dict(sorted(industry_class_B.items(), key=lambda x: x[1], reverse=True))
    industry_class_S_sort_dict = dict(sorted(industry_class_S.items(), key=lambda x: x[1], reverse=True))
    industry_class_B_sort_dict_json_str = json.dumps(industry_class_B_sort_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'industry_class_B_sort_dict.json'), 'w') as json_file:
        json_file.write(industry_class_B_sort_dict_json_str)
    industry_class_S_sort_dict_json_str = json.dumps(industry_class_S_sort_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'industry_class_S_sort_dict.json'), 'w') as json_file:
        json_file.write(industry_class_S_sort_dict_json_str)
    # 写入报错数据
    with open(os.path.join(excel_path, 'data_stat_industry_except.txt'), 'w') as file:
        file.write(str(industry_class_except))


# 获取行业编码
def get_indust_encode(class_industrys, industry):
    for item in class_industrys[0][0][str(3)]:
        try:
            encode = class_industrys[0][0][str(1)][class_industrys[0][0][str(3)].index(item)]
            if item == industry.split("/")[-1] and len(encode) == 4:
                return encode
        except:
            continue


# 划分数据集
def dataset_part(excel_datas_merge):
    industry_dict = {item: [] for item in excel_datas_merge[0][0][1]}
    industry_train_dict = {item: [] for item in excel_datas_merge[0][0][1]}
    industry_valid_dict = {item: [] for item in excel_datas_merge[0][0][1]}
    data_except = []
    # 字典数据写入数据集字典
    for item in range(excel_datas_merge[0][1]):
        if excel_datas_merge[2][0][1][item] == "QYSX_CF" and isinstance(excel_datas_merge[0][0][1][item], str):
            industry_dict[excel_datas_merge[0][0][1][item]].append(excel_datas_merge[0][0][0][item])
        else:
            data_except.append(excel_datas_merge[2][0][1][item])
    # 划分数据集
    # 为保证数据质量，只取行业填选完整的企业，且行业小类至少有70家企业，即占有企业总数量的0.1%及以上
    for key in list(industry_dict.keys()):
        if len(industry_dict[key]) >= 70:
            result = split_list(industry_dict[key], train_ratio, 2)
            industry_train_dict[key] = result[0]
            industry_valid_dict[key] = result[1]
    # 数据集列表
    industry_train_list = [x for sublist in list(industry_train_dict.values()) for x in sublist]
    industry_valid_list = [x for sublist in list(industry_valid_dict.values()) for x in sublist]
    logging.info("dataset_part")
    logging.info(len(data_except))
    # industry_all_list 为 excel_datas_merge[0][0] 中 enterprise 的元素位置
    dataset_part_dict = {"all": industry_train_list + industry_valid_list, "train": industry_train_list, "valid": industry_valid_list}
    dataset_part_dict_json_str = json.dumps(dataset_part_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'dataset_part_dict.json'), 'w') as json_file:
        json_file.write(dataset_part_dict_json_str)
    return


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


# 写入txt文件
def write_all_txt(excel_datas_merge, data_list, indexs):
    rel_list = [
        "industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"
    ]
    with open(os.path.join(excel_path, 'id_md5_dict.json'), 'r') as json_file:
        id_md5_dict = json.loads(str(json_file.read()))
    for index in indexs:
        data_triples = {
            "sub": [],
            "rel": [],
            "obj": [],
        }
        # [0, 1, 2]，list 为读取到的数据在 excel_datas_merge 中的位置
        if rel_list[index] in ["industry", "areacode"]:
            # 企业id——行业
            logging.info("三元组：企业md5——行业、区划")
            for item_index, item in enumerate(excel_datas_merge[index][0][0]):
                if item in data_list:
                    data_triples["sub"].append(id_md5_dict[excel_datas_merge[index][0][0][item_index]])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(excel_datas_merge[index][0][1][item_index])
        elif rel_list[index] == "HW":
            # 企业id——危废类别
            logging.info("三元组：企业md5——危废类别")
            data_total = {}
            data_except = []
            # 存入字典
            for item_index, item in enumerate(excel_datas_merge[index][0][0]):
                if item in data_list:
                    try:
                        sub = excel_datas_merge[index][0][0][item_index]
                        obj = excel_datas_merge[index][0][1][item_index][:4]
                        if sub in data_total:
                            data_total[sub].append(obj)
                        else:
                            data_total[sub] = OrderedSet()
                            data_total[sub].append(obj)
                    except:
                        data_except.append(item)
                        continue
            # 存入三元组
            for key in list(data_total.keys()):
                for data in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(data)
            logging.info(len(data_except))
        elif rel_list[index] == "waste":
            # 企业id——危废代码
            logging.info("三元组：企业md5——危废代码")
            data_total = {}
            data_except = []
            # 存入字典
            for item_index, item in enumerate(excel_datas_merge[index][0][0]):
                if item in data_list:
                    try:
                        sub = excel_datas_merge[index][0][0][item_index]
                        obj = excel_datas_merge[index][0][1][item_index]
                        if sub in data_total:
                            data_total[sub].append(obj)
                        else:
                            data_total[sub] = OrderedSet()
                            data_total[sub].append(obj)
                    except:
                        data_except.append(item)
                        continue
            # 存入三元组
            for key in list(data_total.keys()):
                for data in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(data)
            logging.info(len(data_except))
        elif rel_list[index] in ["material", "product"]:
            # 企业md5——原料、产品
            logging.info("三元组：企业md5——原料、产品")
            data_total = {}
            data_except = []
            dealRawAcc_data = dealRawAcc_BERT_dict(excel_datas_merge[index][0])
            # 存入字典
            for item_index, item in enumerate(dealRawAcc_data[0]):
                if item in data_list:
                    if dealRawAcc_data[1][item_index] != "蚶":
                        try:
                            sub = dealRawAcc_data[0][item_index]
                            obj = dealRawAcc_data[1][item_index]
                            if sub in data_total:
                                data_total[sub].append(obj)
                            else:
                                data_total[sub] = OrderedSet()
                                data_total[sub].append(obj)
                        except:
                            data_except.append(item)
                            continue
            # 存入三元组
            for key in list(data_total.keys()):
                for data in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(data)
            logging.info(len(data_except))
        elif rel_list[index] == "HWwaste":
            # 危废类别——危废代码
            logging.info("三元组：危废类别——危废代码")
            data_total = {}
            data_except = []
            # 存入字典
            for item_index, item in enumerate(excel_datas_merge[index][0][0]):
                try:
                    sub = excel_datas_merge[index][0][0][item_index][:4]
                    obj = excel_datas_merge[index][0][1][item_index]
                    if sub in data_total:
                        data_total[sub].append(obj)
                    else:
                        data_total[sub] = OrderedSet()
                        data_total[sub].append(obj)
                except:
                    data_except.append(item)
                    continue
            # 存入三元组
            for key in list(data_total.keys()):
                for data in data_total[key]:
                    data_triples["sub"].append(key)
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(data)
            logging.info(len(data_except))
        # 导出到文本文件
        pd.DataFrame({0: data_triples["sub"], 1: data_triples["rel"], 2: data_triples["obj"]}, dtype=str).to_csv(
            excel_path + f'/all/{rel_list[index]}.txt', sep='\t', index=False, header=True
        )


def dealRawAcc_BERT_dict(excel_data):
    enterids_ex = []
    rawaccs_ex = []
    data_except = []
    data_item_except = []
    delimiters = ["、", "（", "）"]
    with open(os.path.join(excel_path, 'rawacc_beone_dict.json'), 'r') as json_file:
        rawacc_beone_dict = json.loads(str(json_file.read()))
    for item in range(len(excel_data[list(excel_data.keys())[0]])):
        try:
            # 分割内容
            regex_pattern = '|'.join(map(re.escape, delimiters))
            rawacc_list = re.split(regex_pattern, excel_data[list(excel_data.keys())[1]][item].strip().replace(" ", ""))
            for rawacc in rawacc_list:
                if rawacc:
                    try:
                        rawaccs_ex.append(rawacc_beone_dict[rawacc])
                        enterids_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    except:
                        data_except.append(rawacc)
                        continue
                else:
                    data_item_except.append(excel_data[list(excel_data.keys())[1]][item])
        except:
            data_item_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    logging.info("dealRawAcc_BERT")
    logging.info(len(data_item_except))
    logging.info(data_except)
    return [enterids_ex, rawaccs_ex]


# 读取csv
def read_csv_all(data_path, indexs):
    excel_datas_merge = []
    for index in indexs:
        file_name = data_path + str(index) + ".txt"
        try:
            excel_data = pd.read_csv(file_name, delimiter='\t', encoding="gbk", dtype=str)
        except:
            excel_data = pd.read_csv(file_name, delimiter='\t', encoding="utf-8", dtype=str)
        # 数据字典
        data_dict = {item: [] for item in excel_data.columns}
        # 数据写入字典
        for item in excel_data.index:
            for column in excel_data.columns:
                data_dict[column].append(excel_data[column][item])
        excel_datas_merge.append((data_dict, excel_data.shape[0]))
    logging.info("read excels")
    return excel_datas_merge


def write_train_txt(all_data, dataset_part_dict, id_md5_dict):
    train_data_triples = {
        "sub": [],
        "rel": [],
        "obj": [],
    }
    valid_data_triples = {
        "sub": [],
        "rel": [],
        "obj": [],
    }
    md5_id_dict = dict(zip(id_md5_dict.values(), id_md5_dict.keys()))
    for data in all_data:
        for index, item in enumerate(data[0]['0']):
            try:
                if md5_id_dict[item] in dataset_part_dict["train"]:
                    train_data_triples["sub"].append(data[0]['0'][index])
                    train_data_triples["rel"].append(data[0]['1'][index])
                    train_data_triples["obj"].append(data[0]['2'][index])
                elif md5_id_dict[item] in dataset_part_dict["valid"]:
                    valid_data_triples["sub"].append(data[0]['0'][index])
                    valid_data_triples["rel"].append(data[0]['1'][index])
                    valid_data_triples["obj"].append(data[0]['2'][index])
            except:
                # 危险废物类别——小类代码
                train_data_triples["sub"].append(data[0]['0'][index])
                train_data_triples["rel"].append(data[0]['1'][index])
                train_data_triples["obj"].append(data[0]['2'][index])
                valid_data_triples["sub"].append(data[0]['0'][index])
                valid_data_triples["rel"].append(data[0]['1'][index])
                valid_data_triples["obj"].append(data[0]['2'][index])
    # 导出到文本文件
    pd.DataFrame({0: train_data_triples["sub"], 1: train_data_triples["rel"], 2: train_data_triples["obj"]}, dtype=str).to_csv(
        excel_path + f'/all/train.txt', sep='\t', index=False, header=False
    )
    # 导出到文本文件
    pd.DataFrame({0: valid_data_triples["sub"], 1: valid_data_triples["rel"], 2: valid_data_triples["obj"]}, dtype=str).to_csv(
        excel_path + f'/all/valid.txt', sep='\t', index=False, header=False
    )


if __name__ == '__main__':

    """设置输出日志"""

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    train_ratio = [0.8, 0.2]

    """数据路径"""

    excel_path = r"../../datasets/knowgraph/maxDDD/"

    """获取数据"""

    # # list 为文件名，0,1,2,3,4,5,6,
    # excel_datas = read_csv(excel_path, [
    #     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"
    # ])
    # # 数据整合——全部数据：excel_datas_merge.txt
    # excel_datas_merge = data_merge(excel_datas)

    # 读取全部数据
    with open(os.path.join(excel_path, 'excel_datas_merge.txt'), 'r') as file:
        excel_datas_merge = eval(file.read())

    """数据统计"""

    # # id对应企业名：id_enterprise_dict.json
    # # id对应md5值：id_md5_dict.json
    # data_stat_id_md5(excel_datas_merge)
    # 获取id对应企业名
    with open(os.path.join(excel_path, 'id_enterprise_dict.json'), 'r') as json_file:
        id_enterprise_dict = json.loads(str(json_file.read()))
    # 获取id对应md5值
    with open(os.path.join(excel_path, 'id_md5_dict.json'), 'r') as json_file:
        id_md5_dict = json.loads(str(json_file.read()))

    # # 获取国民经济行业分类数据
    # # list 为文件名，4,5
    # class_industrys = read_csv(excel_path, ["4", "5"])
    # # 行业统计：industry_class_B_sort_dict.json、industry_class_S_sort_dict.json、data_stat_industry_except.txt
    # data_stat_industry(excel_datas_merge, class_industrys)


    """知识图谱及模型训练"""

    # # 所有数据（原料、产品实体链接后）写入 txt，list 为读取到的数据在 excel_datas_merge 中的位置0, 10, 1, 2, 4, 5
    # # 0, 1, 2, 3, 4, 5, 6, 7, 8
    # # "industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"
    # write_all_txt(excel_datas_merge, dataset_part_dict["all"], [0, 3, 4, 5, 6, 7, 8])

    # 划分数据集
    dataset_part(excel_datas_merge)
    # 获取数据集划分
    with open(os.path.join(excel_path, 'dataset_part_dict.json'), 'r') as json_file:
        dataset_part_dict = json.loads(str(json_file.read()))

    # # 训练数据写入 txt
    # # list 为文件名，"industry", "areacode", "HW", "waste", "material", "product", "HWwaste"
    # all_data = read_csv_all(excel_path + "all/", ["industry", "areacode", "HW", "waste", "material", "product", "HWwaste"])
    # write_train_txt(all_data, dataset_part_dict, id_md5_dict)
    # logging.info("Done!!!")
