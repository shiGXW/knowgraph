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
from math import floor
from ordered_set import OrderedSet


# 读取csv
def read_csv(data_path, indexs):
    excel_datas = []
    for index in indexs:
        if index in [7, 8, "0_all", "1_all", "2_all", "4_all", "5_all", "10_all"]:
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
def get_indust_encode(excel_datas, industry):
    for item in excel_datas[7][0][str(3)]:
        try:
            encode = excel_datas[7][0][str(1)][excel_datas[7][0][str(3)].index(item)]
            if item == industry and len(encode) == 4:
                return encode
        except:
            continue


# 原料、产品实体链接
def RawAcc_BEONE(excel_datas):
    # id_enterprise 对应信息字典写入 json
    rawacc_beone_dict = {}
    data_item_except = []
    data_except = []
    delimiters = ["、", "（", "）"]
    # 加载模型
    BBc = BertBaseChinese(excel_path + "bert-base-chinese/", "cuda:1")
    # 加载数据，去重
    excel_data = list(set(excel_datas[4][0]["materialName"] + excel_datas[5][0]["productName"]))
    catalogue_data = excel_datas[8][0][str(5)]
    logging.info(f"excel_data_data：{len(excel_data)}")
    logging.info(f"catalogue_data：{len(catalogue_data)}")
    # 进度条及预估时间
    start_time = time.time()
    current = 0
    total = len(excel_data)
    progress_bar(start_time, total, current)
    for rawacc_mal in excel_data:
        current += 1
        if current % 10 == 0:
            progress_bar(start_time, total, current)

        # 分割内容
        regex_pattern = '|'.join(map(re.escape, delimiters))
        rawacc_list = re.split(regex_pattern, rawacc_mal.strip().replace(" ", ""))
        for rawacc in rawacc_list:
            if rawacc and rawacc not in rawacc_beone_dict:
                rawacc_temp = None
                match_ratio = 0.0

                similarities = BBc.train(
                    [rawacc for _ in range(len(catalogue_data))],
                    excel_datas[8][0][str(5)], batch_size=1024 * 13
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
                data_item_except.append(rawacc_mal)
    rawacc_beone_dict_json_str = json.dumps(rawacc_beone_dict, indent=4, ensure_ascii=False)
    with open(os.path.join(excel_path, 'rawacc_beone_dict.json'), 'w') as json_file:
        json_file.write(rawacc_beone_dict_json_str)


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
    data_except = []
    # 字典数据写入数据集字典
    for item in range(excel_datas[0][1]):
        try:
            if excel_datas[9][0]["entType"][item] == "QYSX_CF":
                industry_dict[excel_datas[0][0][list(excel_datas[0][0].keys())[1]][item]].append(item)
        except:
            data_except.append(excel_datas[9][0]["entType"][item])
            continue
    # 获取企业行业小类及其频次
    with open(os.path.join(excel_path, 'industry_class_S_sort_dict.json'), 'r') as json_file:
        industry_class_S_sort_dict = json.loads(str(json_file.read()))
    # 划分数据集
    # 为保证数据质量，只取行业填选完整的企业，且行业小类至少有两家企业
    for key in list(industry_class_S_sort_dict.keys()):
        if int(industry_class_S_sort_dict[key]) > 1:
            result = split_list(industry_dict[key], train_ratio, 2)
            industry_train_dict[key] = result[0]
            industry_valid_dict[key] = result[1]
    # 数据集列表
    industry_train_list = [x for sublist in list(industry_train_dict.values()) for x in sublist]
    industry_valid_list = [x for sublist in list(industry_valid_dict.values()) for x in sublist]
    logging.info("dataset_part")
    logging.info(len(data_except))
    # industry_all_list 为 excel_datas[0][0] 中 enterprise 的元素位置
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


def dealRawAcc_BERT_dict(excel_datas, excel_data):
    planids_ex = []
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
                        planids_ex.append(excel_data[list(excel_data.keys())[0]][item])
                    except:
                        data_except.append(rawacc)
                        continue
                else:
                    data_item_except.append(excel_data[list(excel_data.keys())[1]][item])
        except:
            data_item_except.append(excel_data[list(excel_data.keys())[1]][item])
            continue
    logging.info("dealRawAcc_Word2Vec")
    logging.info(len(data_item_except))
    logging.info(data_except)
    logging.info("\n")
    return planids_ex, rawaccs_ex


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
                # logging.info(excel_datas[index][0]["planId"].index(planid))
                # logging.info(planid)
                enterpriseId = excel_datas[index][0]["enterpriseId"][excel_datas[index][0]["planId"].index(planid)]
                # logging.info(enterpriseId)
                # enterpriseId 是否在该数据集
                try:
                    if excel_datas[0][0]["entid"].index(enterpriseId) in data_list:
                        sub_list.append(enterpriseId)
                        obj_list.append(planid_data[1][item])
                except:
                    data_except.append(enterpriseId)
                    continue
    logging.info("plan_id_deal")
    logging.info(len(data_except))
    logging.info(len(sub_list), len(obj_list))
    return sub_list, obj_list


# 写入txt文件
def write_all_txt(excel_datas, data_list, indexs):
    rel_list = [
        "industry", "wasteHW", "waste", "pianid", "material", "product", "", "", "", "", "wastecode"
    ]
    for index in indexs:
        with open(os.path.join(excel_path, 'id_md5_dict.json'), 'r') as json_file:
            id_md5_dict = json.loads(str(json_file.read()))
        data_triples = {
            "sub": [],
            "rel": [],
            "obj": [],
        }
        data_total = {}
        # [0, 1, 2]，list 为读取到的数据在 excel_datas 中的位置
        if index in [0]:
            # 企业id——行业
            logging.info("三元组：企业id——行业")
            for item in data_list:
                data_triples["sub"].append(id_md5_dict[excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item]])
                data_triples["rel"].append(rel_list[index])
                data_triples["obj"].append(
                    excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item].replace("\t", "")
                )
        elif index in [1]:
            # 企业id——危废类别
            logging.info("三元组：企业id——危废类别")
            data_except = []
            # 存入字典
            for item in data_list:
                item_index = ent_id_deal(excel_datas, index, item)
                try:
                    sub = excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item_index]
                    obj = excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index].replace("(2016)", "")
                    if sub in data_total:
                        data_total[sub].append(obj)
                    else:
                        data_total[sub] = OrderedSet()
                        data_total[sub].append(obj)
                except:
                    # data_except.append(item)
                    data_except.append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index])
                    continue
            # 存入三元组
            for key in list(data_total.keys()):
                for item in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(item)
            logging.info(len(data_except))
        elif index in [2]:
            # 企业id——危废代码
            logging.info("三元组：企业id——危废代码")
            data_except = []
            for item in data_list:
                item_index = ent_id_deal(excel_datas, index, item)
                try:
                    sub = excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item_index]
                    obj = excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index]
                    if sub in data_total:
                        data_total[sub].append(obj)
                    else:
                        data_total[sub] = OrderedSet()
                        data_total[sub].append(obj)
                except:
                    data_except.append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item_index])
                    continue
            # 存入三元组
            for key in list(data_total.keys()):
                for item in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(item)
            logging.info(len(data_except))
        elif index in [4, 5]:
            # 计划id——原料、产品
            logging.info("三元组：计划id转企业id——原料、产品")
            planid_data = dealRawAcc_BERT_dict(excel_datas, excel_datas[index][0])
            # planid 转 entid
            plan_deal_sub_list, plan_deal_obj_list = plan_id_deal(excel_datas, planid_data, 3, data_list)
            for item in range(len(plan_deal_sub_list)):
                # if data_triples["sub"] and data_triples["rel"] and data_triples["obj"]:
                sub = plan_deal_sub_list[item]
                obj = plan_deal_obj_list[item]
                # obj = plan_deal_obj_list[item].replace("\t", "").replace("?", "")
                if sub in data_total:
                    data_total[sub].append(obj)
                else:
                    data_total[sub] = OrderedSet()
                    data_total[sub].append(obj)
            # 存入三元组
            for key in list(data_total.keys()):
                for item in data_total[key]:
                    data_triples["sub"].append(id_md5_dict[key])
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(item)
        elif index in [10]:
            # 危废类别——危废代码
            logging.info("三元组：危废类别——危废代码")
            data_except = []
            for item in data_list:
                try:
                    sub = excel_datas[index][0][list(excel_datas[index][0].keys())[0]][item].replace("(2016)", "")
                    obj = excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item]
                    if sub in data_total:
                        data_total[sub].append(obj)
                    else:
                        data_total[sub] = OrderedSet()
                        data_total[sub].append(obj)
                except:
                    data_except.append(excel_datas[index][0][list(excel_datas[index][0].keys())[1]][item])
                    continue
            # 存入三元组
            for key in list(data_total.keys()):
                for item in data_total[key]:
                    data_triples["sub"].append(key)
                    data_triples["rel"].append(rel_list[index])
                    data_triples["obj"].append(item)
            logging.info(len(data_except))
        # 导出到文本文件
        pd.DataFrame({0: data_triples["sub"], 1: data_triples["rel"], 2: data_triples["obj"]}, dtype=str).to_csv(
            excel_path + f'/{index}_all.txt', sep='\t', index=False, header=True
        )


def write_train_txt(all_data, dataset_part_dict):
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
    for data in all_data:
        for index, item in enumerate(data[0][0]):
            if item in dataset_part_dict[1]:
                train_data_triples["sub"].append(data[0][0][index])
                train_data_triples["rel"].append(data[0][1][index])
                train_data_triples["obj"].append(data[0][2][index])
            elif item in dataset_part_dict[2]:
                valid_data_triples["sub"].append(data[0][0][index])
                valid_data_triples["rel"].append(data[0][1][index])
                valid_data_triples["obj"].append(data[0][2][index])
            else:
                # 危险废物类别——小类代码
                train_data_triples["sub"].append(data[0][0][index])
                train_data_triples["rel"].append(data[0][1][index])
                train_data_triples["obj"].append(data[0][2][index])
                valid_data_triples["sub"].append(data[0][0][index])
                valid_data_triples["rel"].append(data[0][1][index])
                valid_data_triples["obj"].append(data[0][2][index])


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
    train_ratio = [0.8, 0.2]

    # 获取数据
    excel_path = r"./datasets/knowgraph/"
    # list 为文件名，0,1,2,3,4,5,6,
    excel_datas = read_csv(excel_path, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # # 数据统计——id对应企业名：id_enterprise_dict.json
    # data_stat_id(excel_datas)
    # # 数据统计——id的md5加密：id_md5_dict.json
    # data_stat_md5(excel_datas)
    # # 数据统计——行业统计：industry_class_B_sort_dict.json、industry_class_S_sort_dict.json、data_stat_industry_except.txt
    #
    # data_stat_industry(excel_datas)
    # # 划分数据集
    # dataset_part(excel_datas)
    # # 原料、产品实体链接：rawacc_beone_dict.json
    # RawAcc_BEONE(excel_datas)

    # 知识图谱及模型训练
    # 获取数据集划分
    with open(os.path.join(excel_path, 'dataset_part_dict.json'), 'r') as json_file:
        dataset_part_dict = json.loads(str(json_file.read()))
    # 所有数据写入 txt，list 为读取到的数据在 excel_datas 中的位置0, 10, 1, 2, 4, 5
    write_all_txt(excel_datas, dataset_part_dict["all"], [4, 5])

    # # 训练数据写入 txt
    # # list 为文件名，"0_all", "1_all", "2_all", "4_all", "5_all", "10_all"
    # all_data = read_csv(excel_path, ["0_all", "1_all", "2_all", "4_all", "5_all", "10_all"])
    # write_train_txt(all_data, dataset_part_dict)
    logging.info("Done!!!")
