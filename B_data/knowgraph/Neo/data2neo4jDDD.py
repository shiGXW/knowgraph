import numpy as np
import logging
import pandas as pd
from py2neo import Node, Relationship, Graph, Path, Subgraph
from py2neo import NodeMatcher, RelationshipMatcher

# 连接数据库
# graph = Graph('http://localhost:7474', auth=('neo4j', 'shi@123456'))
# graph = Graph('http://192.168.8.168:7474', auth=('neo4j', 'shi@123456'))
print("连接数据库")
graph = Graph('http://192.168.8.108:7474', auth=('neo4j', 'shi@123456'))


# 读取保存的全部原始数据
def read_all_txt(data_path, indexs, header=None):
    excel_datas_merge = []
    for index in indexs:
        file_name = data_path + str(index) + ".txt"
        try:
            excel_data = pd.read_csv(file_name, delimiter='\t', header=header, encoding="gbk", dtype=str)
        except:
            excel_data = pd.read_csv(file_name, delimiter='\t', header=header, encoding="utf-8", dtype=str)
        # 数据字典
        data_dict = {item: [] for item in excel_data.columns}
        # 数据写入字典
        for item in excel_data.index:
            for column in excel_data.columns:
                data_dict[column].append(excel_data[column][item])
        excel_datas_merge.append((data_dict, excel_data.shape[0]))
    print("read excels")
    return excel_datas_merge


# 节点创建及关系建立
# node1:{"class":类别, "value1":属性1}
def creatrelationship(graph, relationship, node1, node2):
    Node1 = None
    Node2 = None
    # 查找数据库中是否有键值对应的节点
    # 查找节点1
    node1_matcher = NodeMatcher(graph)
    node1s = node1_matcher.match(node1["class"]).where(name=node1["value1"]).all()
    # 查找节点2
    node2_matcher = NodeMatcher(graph)
    node2s = node2_matcher.match(node2["class"]).where(name=node2["value1"]).all()
    if len(node1s) != 0:
        Node1 = node1s[0]
    if len(node2s) != 0:
        Node2 = node2s[0]
    if Node1 is None:
        # 创建节点1
        Node1 = Node(node1["class"], name=node1["value1"])
        graph.create(Node1)
    if Node2 is None:
        # 创建节点2
        Node2 = Node(node2["class"], name=node2["value1"])
        graph.create(Node2)
    rh1 = Relationship(Node1, relationship, Node2)
    graph.create(rh1)


# 数据处理及导入数据库——model
def creatrelationship_deal_model(rel_list, call_dict, all_datas):
    print("导入数据")
    ## 数据匹配及关系建立
    # 1、从txt中获取已有数据
    # 2、数据处理，获取数据中的键值对，重组为新的键值对，存储为list，元素为字典
    # data=[{},{},]
    # 3、以废物代码、性质为例，此时健为废物代码，值为性质
    # # 表格中处理所得数据表头: 废物代码 性质，则重组为新的键值对如下
    # data = [{"code": "271-001-02", "nature": "颜色气味"}, {"code": "271-001-02", "nature": "颜色气味"}]
    # 4、处理data可得node1与node2,循环调用creatrelationship写入数据库
    for datas in all_datas:
        for index, item in enumerate(datas[0][1]):
            rel_temp = rel_list.index(item)
            # print(item.keys())
            # 键为空，键值对不操作
            if pd.isna(datas[0][0][index]):
                continue
            # 值为空，新建键节点
            elif pd.isna(datas[0][2][index]):
                node1 = {"class": call_dict[0][rel_temp], "value1": datas[0][0][index]}
                Node1 = Node(node1["class"], name=node1["value1"])
                graph.create(Node1)
                continue
            # 键值都不为空
            node1 = {"class": call_dict[0][rel_temp], "value1": datas[0][0][index]}
            node2 = {"class": call_dict[1][rel_temp], "value1": datas[0][2][index]}
            creatrelationship(graph, datas[0][1][index], node1, node2)


# 数据处理及导入数据库——enterprise
def creatrelationship_deal_enterprise(begin_poi, end_poi, datas, relationship):
    print("导入数据：" + str(relationship))
    ## 数据匹配及关系建立
    # 1、从txt中获取已有数据
    # 2、数据处理，获取数据中的键值对，重组为新的键值对，存储为list，元素为字典
    # data=[{},{},]
    # 3、以废物代码、性质为例，此时健为废物代码，值为性质
    # # 表格中处理所得数据表头: 废物代码 性质，则重组为新的键值对如下
    # data = [{"code": "271-001-02", "nature": "颜色气味"}, {"code": "271-001-02", "nature": "颜色气味"}]
    # 4、处理data可得node1与node2,循环调用creatrelationship写入数据库
    data_list = []
    for index, item in enumerate(datas[relationship]['0']):
        data_list.append({
            begin_poi: item,
            end_poi: datas[relationship]['2'][index],
        })
    # print(data_list)
    for item in data_list:
        # print(item.keys())
        # 键为空，键值对不操作
        if pd.isna(item[list(item.keys())[0]]):
            continue
        # 值为空，新建键节点
        elif pd.isna(item[list(item.keys())[1]]):
            node1 = {"class": list(item.keys())[0], "value1": item[list(item.keys())[0]]}
            Node1 = Node(node1["class"], name=node1["value1"])
            graph.create(Node1)
            continue
        # 键值都不为空
        node1 = {"class": list(item.keys())[0], "value1": item[list(item.keys())[0]]}
        node2 = {"class": list(item.keys())[1], "value1": item[list(item.keys())[1]]}
        creatrelationship(graph, str(relationship), node1, node2)


# 数据处理及导入数据库
def creatrelationship_deal(begin_poi, end_poi, excelpath, relationship):
    flagio = 0
    print("导入数据：" + relationship)
    ## 数据匹配及关系建立
    # 1、从excel中获取已有数据
    # 2、数据处理，获取数据中的键值对，重组为新的键值对，存储为list，元素为字典
    # data=[{},{},]
    # 3、以废物代码、性质为例，此时健为废物代码，值为性质
    # # 表格中处理所得数据表头: 废物代码 性质，则重组为新的键值对如下
    # data = [{"code": "271-001-02", "nature": "颜色气味"}, {"code": "271-001-02", "nature": "颜色气味"}]
    # 4、处理data可得node1与node2,循环调用creatrelationship写入数据库
    data_excel = pd.read_excel(excelpath, header=0)
    # print(type(data))
    data_arr = data_excel.values
    # print(data_arr)
    data_list = []
    for item in data_arr:
        data_list.append({
            begin_poi: item[0],
            end_poi: item[1],
        })
    # print(data_list)
    for item in data_list:
        # print(item.keys())
        # 键为空，键值对不操作
        if pd.isna(item[list(item.keys())[0]]):
            continue
        # 值为空，新建键节点
        elif pd.isna(item[list(item.keys())[1]]):
            node1 = {"class": list(item.keys())[0], "value1": item[list(item.keys())[0]]}
            Node1 = Node(node1["class"], name=node1["value1"])
            graph.create(Node1)
            continue
        # 键值都不为空
        node1 = {"class": list(item.keys())[0], "value1": item[list(item.keys())[0]]}
        node2 = {"class": list(item.keys())[1], "value1": item[list(item.keys())[1]]}
        # if flagio == 0:
        #     print(node1)
        #     flagio = 1
        # print(node1)
        creatrelationship(graph, relationship, node1, node2)


if __name__ == '__main__':
    # MATCH (n1)-[r]->(n2) RETURN r, n1, n2
    """删除数据库中所有数据"""
    graph.delete_all()

    """数据路径"""
    excel_path = r"../../datasets/knowgraph/maxDDD/"

    """获取数据(模型)"""
    # 关系列表
    rel_list = [
        # md5-行业 md5-企业名称  md5-企业类别  md5-地区划分 md5-危废类别 md5-危废代码 md5-原料 md5-产品 危废代码-危废类别
        "industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"
    ]
    all_datas = read_all_txt(excel_path + "all/", ["valid", "train"], None)

    """数据处理及导入数据库"""
    # 企业相关
    # 数据库中分类
    call_dict = {
        0: ["md5", "md5", "md5", "md5", "md5", "md5", "md5", "md5", "code"],
        1: ["industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"],
    }
    creatrelationship_deal_model(rel_list, call_dict, all_datas)

    print("Done!!!")

    # """获取数据(全部)"""
    # # 关系列表
    # rel_list = [
    #     # md5-行业 md5-企业名称  md5-企业类别  md5-地区划分 md5-危废类别 md5-危废代码 md5-原料 md5-产品 危废代码-危废类别
    #     "industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"
    # ]
    # all_datas = read_all_txt(excel_path + "all/", rel_list, 0)
    #
    # """数据处理及导入数据库"""
    # # 企业相关
    # # 数据库中分类
    # call_dict = {
    #     0: ["md5", "md5", "md5", "md5", "md5", "md5", "md5", "md5", "code"],
    #     1: ["industry", "enterprise", "enttype", "areacode", "HW", "waste", "material", "product", "HWwaste"],
    # }
    # for index, item in enumerate(rel_list):
    #     creatrelationship_deal_enterprise(call_dict[0][index], call_dict[1][index], all_datas[index], index)

    # # 危险废物相关
    # # 数据库中分类
    # # 数据处理及导入数据库1——关系1（危险废物代码、危险废物名称）code——waste_names
    # creatrelationship_deal("code", "waste_names", "./data/1.xlsx", "1")
    # # 数据处理及导入数据库2——关系2（危险废物代码、废物类别）code——waste_category
    # creatrelationship_deal("code", "waste_category", "./data/2.xlsx", "2")
    # # 数据处理及导入数据库3——关系3（危险废物代码、行业来源）code——industry_source
    # creatrelationship_deal("code", "industry_source", "./data/3.xlsx", "3")
    # # # 数据处理及导入数据库4——关系4（危险废物代码、原辅材料）code——raw_materials
    # # creatrelationship_deal("code", "raw_materials", "./data/4.xlsx", "4")
    # # 数据处理及导入数据库5——关系5（危险废物代码、产生环节）code——production_link
    # creatrelationship_deal("code", "production_link", "./data/5.xlsx", "5")
    # # 数据处理及导入数据库6——关系6（危险废物代码、产品）code——product
    # creatrelationship_deal("code", "product", "./data/6.xlsx", "6")
    # # 数据处理及导入数据库7——关系7（危险废物代码、危险特性）code——haz_character
    # creatrelationship_deal("code", "haz_character", "./data/7.xlsx", "7")
    # # # 数据处理及导入数据库8——关系8（危险废物代码、危险废物俗称）code——common_known
    # # creatrelationship_deal("code", "common_known", "./data/8.xlsx", "8")
    # # 数据处理及导入数据库9——关系9（危险废物代码、颜色）code——color
    # creatrelationship_deal("code", "color", "./data/9.xlsx", "9")
    # # 数据处理及导入数据库10——关系10（危险废物代码、状态）code——status
    # creatrelationship_deal("code", "status", "./data/10.xlsx", "10")
    # # 数据处理及导入数据库11——关系11（危险废物代码、气味）code——smell
    # creatrelationship_deal("code", "smell", "./data/11.xlsx", "11")
    # # # 数据处理及导入数据库12——关系12（危险废物代码、PH值）code——PH
    # creatrelationship_deal("code", "PH", "./data/12.xlsx", "12")
    # # 数据处理及导入数据库13——关系13（危险废物代码、水份）code——moisture
    # creatrelationship_deal("code", "moisture", "./data/13.xlsx", "13")
    # # 数据处理及导入数据库14——关系14（危险废物代码、烧失量）code——loss_ignition
    # creatrelationship_deal("code", "loss_ignition", "./data/14.xlsx", "14")
    # # 数据处理及导入数据库15——关系15（危险废物代码、热值）code——calorific_value
    # creatrelationship_deal("code", "calorific_value", "./data/15.xlsx", "15")
