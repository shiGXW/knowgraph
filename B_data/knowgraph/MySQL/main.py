#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql as pymysql


# 唯一编码表
def creatEncodingTable(db):
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 企业唯一编码——企业、企业编码
    sql_Str1 = "create table enterpriseEncoding(" \
              "id int auto_increment primary key," \
              "enterprise varchar(255) not null," \
              "encoding varchar(255) not null" \
              ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 行业唯一编码——行业、行业编码
    sql_Str2 = "create table industryEncoding(" \
              "id int auto_increment primary key," \
              "industry varchar(255) not null," \
              "encoding varchar(255) not null" \
              ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str2)
    db.commit()

    # 用产品唯一编码——用产品、用产品编码
    sql_Str3 = "create table applprodEncoding(" \
               "id int auto_increment primary key," \
               "applprod varchar(255) not null," \
               "encoding varchar(255) not null" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str3)
    db.commit()

    # 工艺唯一编码——工艺、工艺编码
    sql_Str4 = "create table technologyEncoding(" \
               "id int auto_increment primary key," \
               "technology varchar(255) not null," \
               "encoding varchar(255) not null" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str4)
    db.commit()

    # 关键词唯一编码——关键词、关键词编码
    sql_Str5 = "create table keywordEncoding(" \
               "id int auto_increment primary key," \
               "keyword varchar(255) not null," \
               "encoding varchar(255) not null" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str5)
    db.commit()


# 信息对应表（使用编码）
def creatCorresTable(db):
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 企业一对一信息——企业编码、行业编码
    sql_Str1 = "create table enterpriseInfo(" \
              "id int auto_increment primary key," \
              "enterEncoding varchar(255) not null," \
              "industEncoding varchar(255)" \
              ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 企业一对多信息——企业编码、用品编码
    sql_Str1 = "create table enterpriseAppliance(" \
               "id int auto_increment primary key," \
               "enterEncoding varchar(255) not null," \
               "applEncoding varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 企业一对多信息——企业编码、产品编码
    sql_Str1 = "create table enterpriseProduct(" \
               "id int auto_increment primary key," \
               "enterEncoding varchar(255) not null," \
               "prodEncoding varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()

    # 一对多信息——工艺编码、关键词编码
    sql_Str1 = "create table technologyKeyword(" \
               "id int auto_increment primary key," \
               "techEncoding varchar(255) not null," \
               "keyEncoding varchar(255)" \
               ")engine=innoDB default charset=utf8mb3;"
    # print(sql_Str1)
    cursor.execute(sql_Str1)
    db.commit()


def haveTable(db):
    flag = False
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    sql_Str = "SHOW TABLES LIKE 'enterpriseEncoding';"
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
    database = 'enterdata'
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
        creatEncodingTable(db)
        creatCorresTable(db)
    else:
        print("已建表")