#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import os
import re
import json
import pandas as pd
import difflib
from gensim import models


# 读取csv
def read_csv(data_path, indexs):
    excel_datas = []
    for index in indexs:
        if index in [7]:
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


# 数据统计——id对应企业名
def data_stat_id(excel_datas):
    # # 导出到文本文件
    # pd.DataFrame({0: excel_datas[0][0]["entid"], 1: excel_datas[0][0]["indName"]}, dtype=str).to_csv(
    #     excel_path + '/out.txt', sep='\t', index=False, header=False
    # )
    # id_enterprise 对应信息字典写入 json
    id_enterprise_dict = dict(zip(excel_datas[6][0]["entid"], excel_datas[6][0]["entName"]))
    id_enterprise_dict_json_str = json.dumps(id_enterprise_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'id_enterprise_dict.json'), 'w') as json_file:
        json_file.write(id_enterprise_dict_json_str)


# 数据统计——id的md5加密
def data_stat_md5(excel_datas):
    # id 加密 md5
    id_md5 = []
    for id in excel_datas[6][0]["entid"]:
        md5_hash = hashlib.md5()
        md5_hash.update(id.encode("utf-8"))
        id_md5.append(md5_hash.hexdigest())
    # id_md5 对应信息字典写入 json
    id_md5_dict = dict(zip(excel_datas[6][0]["entid"], id_md5))
    id_md5_dict_json_str = json.dumps(id_md5_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'id_md5_dict.json'), 'w') as json_file:
        json_file.write(id_md5_dict_json_str)


# 数据统计——行业统计
def data_stat_industry(excel_datas):
    # 行业大类
    industry_class_B = {}
    # 行业小类
    industry_class_S = {}
    industry_class_except = []
    # 获取行业频次
    for industry in excel_datas[0][0]["indName"]:
        try:
            inencode = get_indust_encode(excel_datas, industry)
            if industry in industry_class_S.keys() and inencode[0:2] in industry_class_B.keys():
                industry_class_S[industry] = industry_class_S[industry] + 1
                industry_class_B[inencode[0:2]] = industry_class_B[inencode[0:2]] + 1
            else:
                industry_class_S[industry] = 1
                industry_class_B[inencode[0:2]] = 1
        except:
            industry_class_except.append(industry)
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
def get_indust_encode(excel_datas, industry):
    for item in excel_datas[7][0][str(3)]:
        try:
            encode = excel_datas[7][0][str(1)][excel_datas[7][0][str(3)].index(item)]
            if item == industry and len(encode) == 4:
                return encode
        except:
            continue


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
    # 获取企业行业小类及其频次
    with open(os.path.join(excel_path, 'industry_class_S_sort_dict.json'), 'r') as json_file:
        industry_class_S_sort_dict = json.loads(str(json_file.read()))
    # 划分数据集
    for key in list(industry_class_S_sort_dict.keys()):
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


# 原辅材料数据处理
def dealRawAcc(excel_data):
    enters_ex = []
    rawaccs_ex = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 读取原辅材料匹配数据
    RawAcc_codes = match_rm_get(excel_path + "匹配度.xlsx")
    for item in range(len(excel_data[list(excel_data.keys())[0]])):
        try:
            # 分割内容
            regex_pattern = '|'.join(map(re.escape, delimiters))
            rawacc_list = re.split(regex_pattern, excel_data[list(excel_data.keys())[1]][item].strip().replace(" ", ""))
            for rawacc in rawacc_list:
                if rawacc:
                    enters_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    rawacc_temp = rawacc
                    if rawacc in RawAcc_codes[:, 0]:
                        rawacc_temp = code_get(RawAcc_codes[list(RawAcc_codes[:, 0]).index(rawacc)])
                    rawaccs_ex.append(rawacc_temp)
                else:
                    data_except.append(excel_data[list(excel_data.keys())[1]][item])
        except:
            data_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    print(data_except)
    return enters_ex, rawaccs_ex


def dealRawAcc_difflib(excel_data):
    enters_ex = []
    rawaccs_ex = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 读取原辅材料匹配数据
    RawAcc_codes = match_rm_get(excel_path + "匹配度.xlsx")
    for item in range(len(excel_data[list(excel_data.keys())[0]])):
        try:
            # 分割内容
            regex_pattern = '|'.join(map(re.escape, delimiters))
            rawacc_list = re.split(regex_pattern, excel_data[list(excel_data.keys())[1]][item].strip().replace(" ", ""))
            for rawacc in rawacc_list:
                if rawacc:
                    enters_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    rawacc_temp = None
                    match_ratio = 0.0
                    for index, RawAcc in enumerate(RawAcc_codes[:, 0]):
                        if difflib.SequenceMatcher(None, rawacc, RawAcc).ratio() > match_ratio:
                            match_ratio = difflib.SequenceMatcher(None, rawacc, RawAcc).ratio()
                            rawacc_temp = code_get(RawAcc_codes[index])
                    rawaccs_ex.append(rawacc_temp)
                else:
                    data_except.append(excel_data[list(excel_data.keys())[1]][item])
        except:
            data_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    print(data_except)
    return enters_ex, rawaccs_ex


def dealRawAcc_Word2Vec(excel_data):
    enters_ex = []
    rawaccs_ex = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 读取原辅材料匹配数据
    RawAcc_codes = match_rm_get(excel_path + "匹配度.xlsx")
    # 加载 wiki 模型
    model = models.word2vec.Word2Vec.load('./datasets/knowgraph/wiki.zh.text.model')
    for item in range(len(excel_data[list(excel_data.keys())[0]])):
        try:
            # 分割内容
            regex_pattern = '|'.join(map(re.escape, delimiters))
            rawacc_list = re.split(regex_pattern, excel_data[list(excel_data.keys())[1]][item].strip().replace(" ", ""))
            for rawacc in rawacc_list:
                if rawacc:
                    enters_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    rawacc_temp = None
                    match_ratio = 0.0
                    for index, RawAcc in enumerate(RawAcc_codes[:0]):
                        if model.wv.similarity(rawacc, RawAcc) > match_ratio:
                            match_ratio = model.wv.similarity(rawacc, RawAcc)
                            rawacc_temp = code_get(RawAcc_codes[index])
                    rawaccs_ex.append(rawacc_temp)
                else:
                    data_except.append(excel_data[list(excel_data.keys())[1]][item])
        except:
            data_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    print(data_except)
    return enters_ex, rawaccs_ex


# 原辅材料匹配数据读取
def match_rm_get(localpath):
    # 读取.xlsx
    print("载入原辅材料匹配数据：" + localpath)
    data = pd.read_excel(localpath, header=None)
    # 材料 [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
    # 编码 [0, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
    codes = data[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]].to_numpy()[1:]
    return codes


# 获取原辅材料匹配数据中对应第一个编码/材料
def code_get(code):
    for item in code:
        # 返回第一个非空编码/材料
        if not pd.isna(item):
            return item


# 写入txt文件
def write_txt(excel_datas, data_list, flag, indexs):
    data_triples = {
        "sub": [],
        "rel": [],
        "obj": [],
    }
    rel_list = ["industry", "wasteHW", "waste", "pianid", "material", "product"]
    for index in indexs:
        # [0, 1, 2]，list 为读取到的数据在 excel_datas 中的位置
        if index == 0:
            for item in data_list:
                if isinstance(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item], str):
                    data_triples["sub"].append(excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(
                        excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item].replace("\t", "").replace("?", "")
                    )
        elif index in [1, 2]:
            for item in data_list:
                item_index = ent_id_deal(excel_datas, index, item)
                if item_index and isinstance(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index], str):
                    data_triples["sub"].append(excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item_index])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(
                        excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index].replace("\t", "").replace("?", "")
                    )
        else:
            planid_data = dealRawAcc_Word2Vec(excel_datas[index][0])
            plan_deal_sub_list, plan_deal_obj_list = plan_id_deal(excel_datas, planid_data, 3, data_list)
            for item in range(len(plan_deal_sub_list)):
                # if data_triples["sub"] and data_triples["rel"] and data_triples["obj"]:
                data_triples["sub"].append(plan_deal_sub_list[item])
                data_triples["rel"].append(rel_list[index])
                data_triples["obj"].append(
                    plan_deal_obj_list[item].replace("\t", "").replace("?", "")
                )

    # 导出到文本文件
    pd.DataFrame({0: data_triples["sub"], 1: data_triples["rel"], 2: data_triples["obj"]}, dtype=str).to_csv(
        excel_path + f'/{flag}_test.txt', sep='\t', index=False, header=False
    )


# 企业id处理，从而获取企业对应于excel_datas中指定数据的index
def ent_id_deal(excel_datas, index, item):
    item_index = False
    ent_id = excel_datas[0][0][list(excel_datas[0][0].keys())[0]][item]
    if ent_id in excel_datas[index][0][list(excel_datas[index][0].keys())[0]]:
        item_index = excel_datas[index][0][list(excel_datas[index][0].keys())[0]].index(ent_id)
    return item_index


# 计划id对应企业id，得到三元组头尾
def plan_id_deal(excel_datas, planid_data, index, data_list):
    sub_list = []
    obj_list = []
    data_except = []
    # 遍历数据
    for item in range(len(planid_data[0])):
        # 查询 planid 对应 enterpriseId
        for planid in excel_datas[index][0]["planId"]:
            if planid == planid_data[0][item]:
                # planid 对应的 enterpriseId
                # print(excel_datas[index][0]["planId"].index(planid))
                # print(planid)
                enterpriseId = excel_datas[index][0]["enterpriseId"][excel_datas[index][0]["planId"].index(planid)]
                # print(enterpriseId)
                # enterpriseId 是否在该数据集
                try:
                    if excel_datas[0][0]["entid"].index(enterpriseId) in data_list:
                        sub_list.append(enterpriseId)
                        obj_list.append(planid_data[1][item])
                except:
                    data_except.append(enterpriseId)
                    continue
    # print(data_except)
    print(len(sub_list), len(obj_list))
    return sub_list, obj_list


if __name__ == '__main__':
    train_ratio = [0.8, 0.2]
    # 获取数据
    excel_path = r"./datasets/knowgraph/"
    # list 为文件名，0,1,2,3,4,5,6,
    excel_datas = read_csv(excel_path, [0, 1, 2, 3, 4, 5, 6, 7])
    # data_stat_id(excel_datas)
    # data_stat_md5(excel_datas)
    data_stat_industry(excel_datas)
    # # 划分数据集
    # industry_train_list, industry_valid_list = dataset_part(excel_datas)
    # # 训练数据写入 txt，list 为读取到的数据在 excel_datas 中的位置
    # write_txt(excel_datas, industry_train_list, "train", [0, 1, 2, 4, 5])
    # write_txt(excel_datas, industry_valid_list, "valid", [0, 1, 2, 4, 5])
    print("Done!!!")
