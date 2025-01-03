#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
from mysql import creatTable, haveTable


def write_HWaste(data, db):
    # print(data)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    data_except = []

    for item in data:
        print(item)
        for fenge in HWastefuzhu2(str(item[10])):
            try:
                sql_Str = "INSERT INTO HWaste(wasteCode, wasteCategory, industrySources, raMaterials, productLink, product, " \
                          "productdetail, excludedProducts, interpro, hwDescription, nhWaste, rhwNames) " \
                          "VALUES ('%s', '%s', '%s','%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (
                                 "NULL" if pd.isna(item[11]) else item[11],
                                 "NULL" if pd.isna(item[0]) else item[0],
                                 "NULL" if pd.isna(item[1]) else item[1],
                                 "NULL" if pd.isna(item[2]) else item[2],
                                 "NULL" if pd.isna(item[3]) else item[3],
                                 "NULL" if pd.isna(item[4]) else item[4],
                                 HWastefuzhu1("NULL" if pd.isna(item[5]) else item[5]),
                                 "NULL" if pd.isna(item[6]) else item[6],
                                 "NULL" if pd.isna(item[7]) else item[7],
                                 "NULL" if pd.isna(item[8]) else item[8],
                                 "NULL" if pd.isna(item[9]) else item[9],
                                 "NULL" if fenge=='nan' else fenge,)
                print(sql_Str)
                # 执行sql语句
                cursor.execute(sql_Str)
                # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
                db.commit()
            except:
                data_except.append(item)
                continue
    print("完成插入")
    return data_except


def HWastefuzhu1(data):
    if data != "NULL":
        if data[0:3] == "见附件":
            data = data[4:]
    return data


def HWastefuzhu2(data):
    if data != "NULL":
        if data.find("；") != -1:
            data = data.split("；")
        else:
            data = data.split("、")
    else:
        data = ["NULL"]
    return data


def write_HWProduct(datadetail, db):
    print(datadetail)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    data_except = []

    for item in datadetail:
        # print(type(item))
        # print(item)
        # print(datadetail["药品类别"][item])
        try:
            sql_Str = "INSERT INTO HWProduct(product, drugCategory) " \
                      "VALUES ('%s', '%s')" % (item[0], item[1])
            print(sql_Str)
            # 执行sql语句
            cursor.execute(sql_Str)
            # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
            db.commit()
        except:
            data_except.append(item)
            continue
    print("完成插入")
    return data_except


def write_HWPcas(datadetailAll, db):
    print(datadetailAll)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    datadetailAll_except = []

    for datadetail in datadetailAll:
        # print(type(datadetail))
        for item in datadetail.index:
            # print(type(item))
            # print(item)
            # print(datadetail["药品类别"][item])
            try:
                sql_Str = "INSERT INTO HWPcas(drugCategory, drugSCategory, drugDCategory, drugName, CAS) " \
                          "VALUES ('%s', '%s','%s', '%s', '%s')" % (
                                  "NULL" if pd.isna(datadetail[1][item]) else datadetail[1][item],
                                  "NULL" if pd.isna(datadetail[2][item]) else datadetail[2][item],
                                  "NULL" if pd.isna(datadetail[3][item]) else datadetail[3][item],
                                  "NULL" if pd.isna(datadetail[4][item]) else datadetail[4][item],
                                  "NULL" if pd.isna(HWPcasfuzhu1(datadetail, item, 5)) else HWPcasfuzhu1(datadetail, item, 5))
                print(sql_Str)
                # 执行sql语句
                cursor.execute(sql_Str)
                # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
                db.commit()
            except:
                datadetailAll_except.append(item)
                continue
    print("完成插入")
    return datadetailAll_except



def HWPcasfuzhu1(datadetail, item, shape):
    try:
        return datadetail[shape][item]
    except:
        return None


def write_HWPCRawAcc(datadetailRM, db):
    # print(datadetailRM)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    data_except = []

    for data in datadetailRM:
        # print(type(datadetail))
        print(len(data.keys()))
        for item in data.index:
            # print(type(item))
            # print(data[1][item])
            for tiaom in range(0, len(data.keys())):
                if not pd.isna(data[tiaom][item]):
                    try:
                        sql_Str = "INSERT INTO hwpcrawacc(drugName, rawacc) " \
                                  "VALUES ('%s', '%s')" % (data[1][item], data[tiaom][item])
                        print(sql_Str)
                        # 执行sql语句
                        cursor.execute(sql_Str)
                        # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
                        db.commit()
                    except:
                        data_except.append(item)
                        continue
    print("完成插入")
    return data_except


def write_enterpriseInfo(data, datalen, db):
    # print(datadetailRM)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    data_except = []

    for item in range(datalen):
        # print(data[list(data.keys())[0]][item])
        try:
            sql_Str = "INSERT INTO enterpriseInfo(enterprise, industry, industryID) " \
                      "VALUES ('%s', '%s', '%s')" % (
                data[list(data.keys())[0]][item], data[list(data.keys())[1]][item], data[list(data.keys())[2]][item]
            )
            print(sql_Str)
            # 执行sql语句
            cursor.execute(sql_Str)
            # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
            db.commit()
        except:
            data_except.append(sql_Str)
            continue
    print("完成插入")
    return data_except

def write_enterpriseappliance(data, datalen, db):
    # print(datadetailRM)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    data_except = []

    for item in range(datalen):
        # print(data[list(data.keys())[0]][item])
        try:
            for key in data[list(data.keys())[3]][item].split("、"):
                    sql_Str = "INSERT INTO enterpriseappliance(enterprise, Ematerial) " \
                              "VALUES ('%s', '%s')" % (
                        data[list(data.keys())[0]][item], key
                    )
                    # print(sql_Str)
                    # 执行sql语句
                    cursor.execute(sql_Str)
                    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
                    db.commit()
        except:
            data_except.append(sql_Str)
            continue
    print("完成插入")
    return data_except


def write_enterpriseproduct(data, datalen, db):
    # print(datadetailRM)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    data_except = []

    for item in range(datalen):
        # print(data[list(data.keys())[0]][item])
        try:
            for key in data[list(data.keys())[4]][item].split("、"):
                    sql_Str = "INSERT INTO enterpriseproduct(enterprise, Eproduct) " \
                              "VALUES ('%s', '%s')" % (
                        data[list(data.keys())[0]][item], key
                    )
                    # print(sql_Str)
                    # 执行sql语句
                    cursor.execute(sql_Str)
                    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
                    db.commit()
        except:
            data_except.append(sql_Str)
            continue
    print("完成插入")
    return data_except

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
    #     data, datadetail, datadetailAll = getAlldata(HWpath, dir, files)
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
    # # 获取企业数据
    # enterpath = r"./data/enter/企业数据.xlsx"
    # dataenter = pd.read_excel(enterpath, header=None)
    # dataenterdict = {
    #     "enterprise": [],
    #     "industry": [],
    #     "industryID": [],
    #     "Ematerial": [],
    #     "Eproduct": [],
    #     "Ekeyword": [],
    #     "Etechnology": [],
    # }
    # for item in dataenter.index:
    #     tiaom = 0
    #     for key in dataenterdict.keys():
    #         # print(dataenter[tiaom][item])
    #         dataenterdict[key].append(dataenter[tiaom][item])
    #         tiaom += 1
    # # data_except5 = write_enterpriseInfo(dataenterdict, dataenter.shape[0], db)
    # # print(data_except5)
    # data_except6 = write_enterpriseappliance(dataenterdict, dataenter.shape[0], db)
    # print(data_except6)
    # data_except7 = write_enterpriseproduct(dataenterdict, dataenter.shape[0], db)
    # print(data_except7)
