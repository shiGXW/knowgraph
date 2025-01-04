#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from mysqlIn import *


def dirs_name(localpath):
    # root 当前目录路径（包含所有子目录）
    # dirs 当前路径下所有子目录（同一路径下的存一个列表中）
    # files 当前路径下所有非目录子文件（同一路径下的存一个列表中）
    # print(os.walk(localpath))
    for root, dirs, files in os.walk(localpath):
        return dirs


def file_name(file_dir):
    fileList = []
    # root 当前目录路径（包含所有子目录）
    # dirs 当前路径下所有子目录（同一路径下的存一个列表中）
    # files 当前路径下所有非目录子文件（同一路径下的存一个列表中）
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # print(file)
            # os.path.splitext()函数将路径拆分为文件名+扩展名
            # 或
            # if file.split('.')[1].strip() == 'txt':
            if os.path.splitext(file)[1] == '.xlsx':
                fileList.append(file)
    return fileList


def getHWdata(localpath, dir, files):
    dataAll = None
    datadetailAll = []
    datadetail = []
    datadetaile = []
    # 获取数据
    for item in files:
        if item[0:2] == 'HW':
            # 读取HW02医药废物.xlsx
            # print(item)
            filename = localpath + dir + "/" + item
            print("载入文件：" + filename)
            data = pd.read_excel(filename, header=0)
            # 填充空值
            data['废物类别'] = data['废物类别'].fillna(method='pad', axis=0)
            data['行业来源'] = data['行业来源'].fillna(method='pad', axis=0)
            if dataAll is None:
                dataAll = data.values
            else:
                dataAll = np.concatenate((dataAll, data.values), axis=0)

        if item[0:2] == '附件':
            # 读取附件
            datadetail.append(item)
            # print(cell[3:4])
            # datadetaile.append(cell[3:])
            filename = localpath + dir + "/" + item
            print("载入文件：" + filename)
            datafu = pd.read_excel(filename, header=None)
            # print(data.loc[data['产品明细'] == cell]["产品"].values[0])
            # print(type(data.loc[data['产品明细'] == cell]["产品"]))
            for timm in datafu[1]:
                if not pd.isna(timm):
                    datadetail_temp = [item[3:].replace(".xlsx", ""), timm]
                    datadetaile.append(datadetail_temp)

            datakeys = datafu.keys()
            # print(datakeys)
            datafu[datakeys[0]] = datafu[datakeys[0]].fillna(method='pad', axis=0)
            datafu[datakeys[1]] = datafu[datakeys[1]].fillna(method='pad', axis=0)
            datafu[datakeys[2]] = datafu[datakeys[2]].fillna(method='pad', axis=0)
            datafu[datakeys[3]] = datafu[datakeys[3]].fillna(method='pad', axis=0)
            datadetailAll.append(datafu)
            # if datadetailAll is None:
            #     datadetailAll = data.values
            # else:
            #     datadetailAll = np.concatenate((datadetailAll, data.values), axis=0)
        # print(data.values.shape)
    # data.values 是pandas读取到的数据（二维数组） data.values[0][0]：表示第0行第0列
    return dataAll, datadetaile, datadetailAll


if __name__ == '__main__':
    # utf8mb3, utf8_general_ci
    host = 'localhost'
    user = 'root'
    password = 'shi@123456'
    database = 'gcndata'
    charset = 'utf8mb3'
    db = pymysql.connect(host=host,
                         user=user,
                         password=password,
                         database=database,
                         charset=charset)
    print("数据库连接成功！")
    # 数据库是否已存在表
    if not haveTable(db):
        print("表不存在，建表中......")
        creatTable(db)
    else:
        print("已建表")
    # # 获取危险废物数据
    # HWpath = r"./data/HWdata/"
    # dirs = dirs_name(HWpath)
    # print("目录：", dirs)
    # for dir in dirs:
    #     files = file_name(HWpath + dir)
    #     print("文件：", files)
    #     data, datadetail, datadetailAll = getHWdata(HWpath, dir, files)
    #     # print(datadetailAll)
    #     # # print(datadetailAll[0].keys())
    #     # print(data)
    #     # print(datadetail)
    #     # 数据写入数据库
    #     data_except1 = write_HWaste(data, db)
    #     print(data_except1)
    #     data_except2 = write_HWProduct(datadetail, db)
    #     print(data_except2)
    #     data_except3 = write_HWPcas(datadetailAll, db)
    #     print(data_except3)
    # # 获取原辅数据
    # rawaccpath = r"./data/RawAcc/"
    # files = file_name(rawaccpath)
    # print(files)
    # datadetailRM = []
    # for item in files:
    #     if item[0:2] == '附件':
    #         filename = rawaccpath + item
    #         print("载入文件：" + filename)
    #         dataRM = pd.read_excel(filename, header=None)
    #         datadetailRM.append(dataRM)
    # data_except4 = write_HWPCRawAcc(datadetailRM, db)
    # print(data_except4)
    # 获取企业数据
    enterpath = r"./data/enter/企业数据.xlsx"
    dataenter = pd.read_excel(enterpath, header=None)
    dataenterdict = {
        "enterprise": [],
        "industry": [],
        "industryID": [],
        "Ematerial": [],
        "Eproduct": [],
        "Ekeyword": [],
        "Etechnology": [],
    }
    for item in dataenter.index:
        tiaom = 0
        for key in dataenterdict.keys():
            # print(dataenter[tiaom][item])
            dataenterdict[key].append(dataenter[tiaom][item])
            tiaom += 1
    # data_except5 = write_enterpriseInfo(dataenterdict, dataenter.shape[0], db)
    # print(data_except5)
    # 导出到文本文件
    pd.DataFrame({0: dataenterdict["enterprise"], 1: dataenterdict["industry"]}).to_csv('./data/industry.txt', sep='\t', index=False)

    # data_except6 = write_enterpriseappliance(dataenterdict, dataenter.shape[0], db)
    # print(data_except6)
    # data_except7 = write_enterpriseproduct(dataenterdict, dataenter.shape[0], db)
    # print(data_except7)

    pd.DataFrame({0: dataenterdict["enterprise"], 1: dataenterdict["Ematerial"]}).to_csv('./data/material.txt', sep='\t', index=False)
    pd.DataFrame({0: dataenterdict["enterprise"], 1: dataenterdict["Eproduct"]}).to_csv('./data/product.txt', sep='\t', index=False)