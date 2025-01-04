# -*-coding:utf-8-*-
import json
import re
import networkx as nx
from orderedset import OrderedSet
from networkx.algorithms import community
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

'''
    使用算法生成知识图谱的x、y坐标
'''


def load_konwgraph_data(train_path, test_path):
    nodes_set = OrderedSet()

    for line in open(train_path):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        nodes_set.add(sub)
        nodes_set.add(obj)
    for line in open(test_path):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        nodes_set.add(sub)
        nodes_set.add(obj)
    nodes = list(nodes_set)
    edges = []
    for line in open(train_path):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        edges.append({
            "source": sub,
            "target": obj
        })
    for line in open(test_path):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        edges.append({
            "source": sub,
            "target": obj
        })

    return nodes, edges


def get_laidu_networkx_index(nodes, edges):
    # 获取每个实体分配坐标
    G = nx.Graph()
    # 创建一个无向图并添加节点和边
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    '''计算节点的位置'''

    # # 圆形布局（Circular layout）
    # # 将所有节点围绕中心按照一定的顺序排列成圆形，节点之间的距离相等。
    # pos = nx.circular_layout(G)

    # # 随机布局（Randomlayout）
    # # 随机分配每个节点的位置。
    # pos = nx.random_layout(G)

    # 调整布局（Kamada-Kawai layout）
    # 基于图的距离矩阵，通过最小化连线的总长度来确定每个节点的位置。
    pos = nx.kamada_kawai_layout(G)

    # # 力导向布局（Fruchterman-Reingold layout）
    # # 节点之间的连线看作弹簧，节点之间的距离保持平衡，节点和图形边缘之间的距离保持恒定。
    # pos = nx.spring_layout(G, k=1, iterations=50)

    # pos 是一个字典，其键是节点，值是对应的 (x, y) 坐标
    # 你可以直接访问这些坐标
    x_coords = {node: pos[node][0] for node in G.nodes()}
    y_coords = {node: pos[node][1] for node in G.nodes()}

    x_y_dict = {}
    x_y_arr = []
    for key in nodes:
        x_y_obj = {
            'id': len(x_y_arr),
            'x': x_coords.get(key),
            'y': y_coords.get(key),
        }
        x_y_dict[key] = x_y_obj
        x_y_arr.append([x_coords.get(key), y_coords.get(key)])
    # 打印坐标
    print("X Coordinates:", x_coords)
    print("Y Coordinates:", y_coords)

    return x_y_dict


def Data2Web(train_path, valid_path):
    nodes_set = OrderedSet()
    nodes_cat = []
    data_web = {
        "nodes": [],
        "links": [],
        "categories": [
            {"name": "enterprise"},
            {"name": "industry"},
            {"name": "areacode"},
            {"name": "MP"},
            {"name": "HW"},
            {"name": "waste"},
            {"name": "HWwaste"},
        ]
    }
    # 确定点位id及点位类别
    # for line in open(train_path):
    #     sub, rel, obj = map(str, line.strip().split('\t'))
    #     # 验证是否是md5格式，以确定其为企业id
    #     if re.findall(r"([a-fA-F\d]{32})", sub) and len(sub) == 32 and sub not in nodes_set:
    #         nodes_set.add(sub)
    #         nodes_cat.append("enterprise")
    #     elif sub.startswith("HW") and len(sub) == 4 and sub not in nodes_set:
    #         nodes_set.add(sub)
    #         nodes_cat.append("HW")
    #     if obj not in nodes_set:
    #         if rel == "material" or rel == "product":
    #             nodes_set.add(obj)
    #             nodes_cat.append("MP")
    #         else:
    #             nodes_set.add(obj)
    #             nodes_cat.append(rel)
    for line in open(valid_path):
        sub, rel, obj = map(str, line.strip().split('\t'))
        # 验证是否是md5格式，以确定其为企业id
        if re.findall(r"([a-fA-F\d]{32})", sub) and len(sub) == 32 and sub not in nodes_set:
            nodes_set.add(sub)
            nodes_cat.append("enterprise")
        elif sub.startswith("HW") and len(sub) == 4 and sub not in nodes_set:
            nodes_set.add(sub)
            nodes_cat.append("HW")
        if obj not in nodes_set:
            if rel == "material" or rel == "product":
                nodes_set.add(obj)
                nodes_cat.append("MP")
            else:
                nodes_set.add(obj)
                nodes_cat.append(rel)
    nodes_list = list(nodes_set)
    # 构建links
    # for line in open(train_path):
    #     sub, rel, obj = map(str, line.strip().split('\t'))
    #     data_web["links"].append({
    #         "source": str(nodes_list.index(sub)),
    #         "target": str(nodes_list.index(obj)),
    #     })
    for line in open(valid_path):
        sub, rel, obj = map(str, line.strip().split('\t'))
        data_web["links"].append({
            "source": str(nodes_list.index(sub)),
            "target": str(nodes_list.index(obj)),
        })
    # 构建nodes
    category_list = [value for dict in data_web["categories"] for value in dict.values()]
    for index, item in enumerate(nodes_cat):
        # 企业节点
        if item == "enterprise":
            data_web["nodes"].append({
                "id": index,
                "name": nodes_list[index],
                "symbolSize": 10,
                "category": category_list.index(item)
            })
        else:
            data_web["nodes"].append({
                "id": index,
                "name": nodes_list[index],
                "symbolSize": 50,
                "category": category_list.index(item)
            })
    print(data_web)
    # 打开文件并写入json数据
    with open("../datasets/knowgraph/max/all/data-web1.json", "w") as file:
        json.dump(data_web, file)


if __name__ == '__main__':
    data_path = r"../../datasets/knowgraph/maxDDD/all/"
    # nodes, edges = load_konwgraph_data(data_path + "train.txt", data_path + "valid.txt")
    # coordinates, x_y_dict = get_laidu_networkx_index(nodes, edges)
    # print(coordinates)
    # print(x_y_dict)
    Data2Web(data_path + "train.txt", data_path + "valid.txt")

