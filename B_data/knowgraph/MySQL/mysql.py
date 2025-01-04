#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql as pymysql


def creatTable(db):
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 企业一对多信息——企业、原料
    sql_Str1 = "create table enterpriseAppliance(" \
               "id int auto_increment primary key," \
               "enterprise varchar(255) not null," \
               "Ematerial varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 企业一对多信息——企业、产品
    sql_Str1 = "create table enterpriseProduct(" \
               "id int auto_increment primary key," \
               "enterprise varchar(255) not null," \
               "Eproduct varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 企业一对多信息——企业、工艺关键词
    sql_Str1 = "create table enterpriseProduct(" \
               "id int auto_increment primary key," \
               "enterprise varchar(255) not null," \
               "Ekeyword varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 企业唯一编码——企业、企业编码
    sql_Str1 = "create table enterpriseInfo(" \
              "id int auto_increment primary key," \
              "enterprise varchar(255) not null," \
              "industry varchar(255) not null," \
              "industryID varchar(255) not null" \
              ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    # "Etechnology varchar(255) not null" \
    cursor.execute(sql_Str1)
    db.commit()


    # 危险废物Hazardous waste
    sql_Str = "create table HWaste(" \
              "id int auto_increment primary key," \
              "wasteCode varchar(255) not null," \
              "wasteCategory varchar(255)," \
              "industrySources varchar(255)," \
              "raMaterials varchar(255)," \
              "productLink varchar(255)," \
              "product varchar(255)," \
              "productdetail varchar(255)," \
              "excludedProducts varchar(255)," \
              "interpro varchar(255)," \
              "hwDescription varchar(255)," \
              "nhWaste varchar(255)," \
              "rhwNames varchar(255)" \
              ")engine=innoDB default charset=utf8mb4;"
    cursor.execute(sql_Str)
    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    db.commit()

    # 危险废物对应产品
    sql_Str = "create table HWProduct(" \
              "id int auto_increment primary key," \
              "product varchar(255) not null," \
              "drugCategory varchar(255) not null" \
              ")engine=innoDB default charset=utf8mb4;"
    cursor.execute(sql_Str)
    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    db.commit()

    # 产品对应品名及CAS号
    sql_Str = "create table HWPcas(" \
              "id int auto_increment primary key," \
              "drugCategory varchar(255) not null," \
              "drugSCategory varchar(255)," \
              "drugDCategory varchar(255)," \
              "drugName varchar(255) not null," \
              "CAS varchar(255)" \
              ")engine=innoDB default charset=utf8mb4;"
    cursor.execute(sql_Str)
    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    db.commit()

    # 品名及CAS号对应原辅料
    sql_Str = "create table HWPCRawAcc(" \
              "id int auto_increment primary key," \
              "drugName varchar(255) not null," \
              "rawacc varchar(255)" \
              ")engine=innoDB default charset=utf8mb4;"
    cursor.execute(sql_Str)
    # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    db.commit()

    # # 产品特性
    # sql_Str = "create table procharacter(" \
    #           "id int auto_increment primary key," \
    #           "wasteCode varchar(255) not null," \
    #           "proname varchar(255)," \
    #           "character varchar(255)" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()
    #
    # # 废物特性
    # sql_Str = "create table hwcharacter(" \
    #           "id int auto_increment primary key," \
    #           "wasteCode varchar(255) not null," \
    #           "hwname varchar(255)," \
    #           "character varchar(255)" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()
    #
    # # 行业来源1
    # sql_Str = "create table industrysource1(" \
    #           "id int auto_increment primary key," \
    #           "cat varchar(255) not null," \
    #           "catname varchar(255) not null," \
    #           "descript varchar(255) not null" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()
    #
    # # 行业来源2
    # sql_Str = "create table industrysource2(" \
    #           "id int auto_increment primary key," \
    #           "cat varchar(255) not null," \
    #           "category varchar(255) not null," \
    #           "isname varchar(255) not null," \
    #           "descript varchar(255) not null" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()
    #
    # # 行业来源3
    # sql_Str = "create table industrysource3(" \
    #           "id int auto_increment primary key," \
    #           "category varchar(255) not null," \
    #           "class varchar(255) not null," \
    #           "isname varchar(255) not null," \
    #           "descript varchar(255) not null" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()
    #
    # # 行业来源4
    # sql_Str = "create table uram(" \
    #           "id int auto_increment primary key," \
    #           "class varchar(255) not null," \
    #           "isdetail varchar(255) not null," \
    #           "isname varchar(255) not null," \
    #           "ydescribe varchar(255) not null," \
    #           "ndescribe varchar(255)" \
    #           ")engine=innoDB default charset=utf8mb4;"
    # cursor.execute(sql_Str)
    # # 提交当前事务，才会提交到数据库，可以尝试只执行上面的代码，看看结果
    # db.commit()


def haveTable(db):
    flag = False
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    sql_Str = "SHOW TABLES LIKE 'enterpriseInfo';"
    # print(sql_Str)
    cursor.execute(sql_Str)
    result = cursor.fetchall()
    if result:
        flag = True
    return flag


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