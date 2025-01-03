import numpy as np
import pandas as pd
from py2neo import Node, Relationship, Graph, Path, Subgraph
from py2neo import NodeMatcher, RelationshipMatcher

# 连接数据库
graph = Graph('http://localhost:7474', auth=('neo4j', 'shi@123456'))


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
    # 删除数据库中所有数据
    graph.delete_all()

    # 数据处理及导入数据库
    creatrelationship_deal("enterEncoding", "industEncoding", "./data/1.xlsx", "1")

    # # 数据处理及导入数据库1——关系1（危险废物代码、危险废物名称）code——waste_names
    # creatrelationship_deal("code", "waste_names", "./data/1.xlsx", "1")
    # # 数据处理及导入数据库5——关系5（危险废物代码、产生环节）code——production_link
    # creatrelationship_deal("code", "production_link", "./data/5.xlsx", "5")
    # # 数据处理及导入数据库6——关系6（危险废物代码、产品）code——product
    # creatrelationship_deal("code", "product", "./data/6.xlsx", "6")
    # # 数据处理及导入数据库16——关系16（危险废物代码、企业名称）code——enterprise_name
    # creatrelationship_deal("code", "enterprise", "./data/16.xlsx", "16")
    # # 数据处理及导入数据库17——关系17（企业名称、原辅材料）enterprise——raw_materials
    # creatrelationship_deal("enterprise", "raw_materials", "./data/17.xlsx", "17")
    # # 数据处理及导入数据库18——关系18（企业名称、工艺）enterprise——technology
    # creatrelationship_deal("enterprise", "technology", "./data/18.xlsx", "18")
    # # 数据处理及导入数据库19——关系19（企业名称、产品）enterprise——product
    # creatrelationship_deal("enterprise", "product", "./data/19.xlsx", "19")
    # # 数据处理及导入数据库20——关系20（企业、行业类别）enterprise——industry_category
    # creatrelationship_deal("enterprise", "industry_category", "./data/20.xlsx", "20")
    # # # 数据处理及导入数据库21——关系21（危险废物代码、俗称）code——common_name
    # # creatrelationship_deal("code", "common_name", "./data/21.xlsx", "21")



    # 数据处理及导入数据库1——关系1（危险废物代码、危险废物名称）code——waste_names
    creatrelationship_deal("code", "waste_names", "./data/1.xlsx", "1")
    # 数据处理及导入数据库2——关系2（危险废物代码、废物类别）code——waste_category
    creatrelationship_deal("code", "waste_category", "./data/2.xlsx", "2")
    # 数据处理及导入数据库3——关系3（危险废物代码、行业来源）code——industry_source
    creatrelationship_deal("code", "industry_source", "./data/3.xlsx", "3")
    # # 数据处理及导入数据库4——关系4（危险废物代码、原辅材料）code——raw_materials
    # creatrelationship_deal("code", "raw_materials", "./data/4.xlsx", "4")
    # 数据处理及导入数据库5——关系5（危险废物代码、产生环节）code——production_link
    creatrelationship_deal("code", "production_link", "./data/5.xlsx", "5")
    # 数据处理及导入数据库6——关系6（危险废物代码、产品）code——product
    creatrelationship_deal("code", "product", "./data/6.xlsx", "6")
    # 数据处理及导入数据库7——关系7（危险废物代码、危险特性）code——haz_character
    creatrelationship_deal("code", "haz_character", "./data/7.xlsx", "7")
    # # 数据处理及导入数据库8——关系8（危险废物代码、危险废物俗称）code——common_known
    # creatrelationship_deal("code", "common_known", "./data/8.xlsx", "8")
    # 数据处理及导入数据库9——关系9（危险废物代码、颜色）code——color
    creatrelationship_deal("code", "color", "./data/9.xlsx", "9")
    # 数据处理及导入数据库10——关系10（危险废物代码、状态）code——status
    creatrelationship_deal("code", "status", "./data/10.xlsx", "10")
    # 数据处理及导入数据库11——关系11（危险废物代码、气味）code——smell
    creatrelationship_deal("code", "smell", "./data/11.xlsx", "11")

    # # 数据处理及导入数据库12——关系12（危险废物代码、PH值）code——PH
    # creatrelationship_deal("code", "PH", "./data/12.xlsx", "12")
    # # 数据处理及导入数据库13——关系13（危险废物代码、水份）code——moisture
    # creatrelationship_deal("code", "moisture", "./data/13.xlsx", "13")
    # # 数据处理及导入数据库14——关系14（危险废物代码、烧失量）code——loss_ignition
    # creatrelationship_deal("code", "loss_ignition", "./data/14.xlsx", "14")
    # # 数据处理及导入数据库15——关系15（危险废物代码、热值）code——calorific_value
    # creatrelationship_deal("code", "calorific_value", "./data/15.xlsx", "15")

    # 数据处理及导入数据库16——关系16（危险废物代码、企业名称）code——enterprise_name
    creatrelationship_deal("code", "enterprise_name", "./data/16.xlsx", "16")
    # 数据处理及导入数据库17——关系17（企业名称、原辅材料）enterprise——raw_materials
    creatrelationship_deal("enterprise_name", "raw_materials", "./data/17.xlsx", "17")

    # # 数据处理及导入数据库18——关系18（企业名称、工艺）enterprise——technology
    # creatrelationship_deal("enterprise_name", "technology", "./data/18.xlsx", "18")

    # 数据处理及导入数据库19——关系19（企业名称、产品）enterprise——product
    creatrelationship_deal("enterprise_name", "product", "./data/19.xlsx", "19")


    # # 数据处理及导入数据库1——关系1（企业、企业性质）enterprise——enterprise_nature
    # creatrelationship_deal("enterprise", "enterprise_nature", "./data/1.xlsx", "1")
    # # 数据处理及导入数据库2——关系2（企业、行业类别）enterprise——industry_category
    # creatrelationship_deal("enterprise", "industry_category", "./data/2.xlsx", "2")
    # # 数据处理及导入数据库3——关系3（企业、原料）enterprise——raw_materials
    # creatrelationship_deal("enterprise", "raw_materials", "./data/3.xlsx", "3")
    # # 数据处理及导入数据库4——关系4（企业、产品）enterprise——product
    # creatrelationship_deal("enterprise", "product", "./data/4.xlsx", "4")
    # # 数据处理及导入数据库5——关系5 (行业、行业代码）industry——industry_code
    # creatrelationship_deal("industry", "industry_code", "./data/5.xlsx", "5")
    # # 数据处理及导入数据库6——关系6（危险废物代码、名录危废名称）code——waste_names
    # creatrelationship_deal("code", "waste_names", "./data/6.xlsx", "6")
    # # 数据处理及导入数据库7——关系7（危险废物代码、危险特性）code——haz_character
    # creatrelationship_deal("code", "haz_character", "./data/7.xlsx", "7")
    # # 数据处理及导入数据库8——关系8（危险废物代码、行业类别）code——industry_category
    # creatrelationship_deal("code", "industry_category", "./data/8.xlsx", "8")
    # # 数据处理及导入数据库9——关系9（危险废物代码、废物类别）code——waste_category
    # creatrelationship_deal("code", "waste_category", "./data/9.xlsx", "9")
    # # 数据处理及导入数据库10——关系10（危险废物代码、废物描述）code——waste_descript
    # creatrelationship_deal("code", "waste_descript", "./data/10.xlsx", "10")
    # # 数据处理及导入数据库11——关系11（危险废物代码、企业名称）code——enterprise_name
    # creatrelationship_deal("code", "enterprise_name", "./data/11.xlsx", "11")


    # sdfsd = [['193-001-21', 'HW21含铬废物'],
    #          ['193-001-21', '毛皮鞣制及制品加工'],
    #          ['193-001-21', '铬鞣剂'],
    #          ['193-001-21', '铬鞣、复鞣过程'],
    #          ['193-001-21', '废水处理污泥、残渣'],
    #          ['193-001-21', '含铬污泥'],
    #          ['193-001-21', '白色'],
    #          ['193-001-21', '固体'],
    #          ['193-001-21', '刺激气味'],
    #          ['193-001-21', '故城县鸿利皮草有限公司'],
    #          ['故城县鸿利皮草有限公司', '毛皮鞣制加工'],
    #          ['故城县鸿利皮草有限公司', '1931'],
    #          ['故城县鸿利皮草有限公司', '鞣剂、踢皮油、貂皮'],
    #          ['故城县鸿利皮草有限公司', '分路-浸水-脱脂-酶软化-浸酸-鞣制-水洗-晾干-质检-入库'],
    #          ['故城县鸿利皮草有限公司', '熟水貂皮']]
    # sdfsd = [['193-001-21', 'HW21含铬废物'],
    #          ['193-001-21', '毛皮鞣制及制品加工'],
    #          ['193-001-21', '铬鞣剂'],
    #          ['193-001-21', '铬鞣、复鞣过程'],
    #          ['193-001-21', '废水处理污泥、残渣'],
    #          ['193-001-21', '含铬污泥'],
    #          ['193-001-21', '白色'],
    #          ['193-001-21', '固体'],
    #          ['193-001-21', '刺激气味'],
    #          ['193-001-21', '   县   皮草有限公司'],
    #          ['   县   皮草有限公司', '毛皮鞣制加工'],
    #          ['   县   皮草有限公司', '1931'],
    #          ['   县   皮草有限公司', '鞣剂、踢皮油、貂皮'],
    #          ['   县   皮草有限公司', '分路-浸水-脱脂-酶软化-浸酸-鞣制-水洗-晾干-质检-入库'],
    #          ['   县   皮草有限公司', '熟水貂皮']]
    #
    # print(sdfsd)
    #
    # nodes = [
    #     {
    #         'class': '危险废物代码',
    #         'value1': '193-001-21'
    #     },
    #     {
    #         'class': '废物名称',
    #         'value1': '废水处理污泥、残渣'
    #     },
    #     {
    #         'class': '废物类别',
    #         'value1': 'HW21含铬废物'
    #     },
    #     {
    #         'class': '行业来源',
    #         'value1': '毛皮鞣制剂制品加工'
    #     },
    # ]
    # edge = [
    #     {
    #         'start': '193-001-21',
    #         'goal': '废水处理污泥、残渣',
    #         'type': '1'
    #     },
    #     {
    #         'start': '193-001-21',
    #         'goal': 'HW21含铬废物',
    #         'type': '2'
    #     }
    # ]
    # for item in nodes:
    #     print(item)
