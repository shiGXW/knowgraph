from py2neo import Graph

# 连接到Neo4j数据库
graph = Graph("http://localhost:7474", auth=('neo4j', 'shi@123456'))

# 测试
def test(graph):

    # Cypher查询语句
    query = "MATCH (n) RETURN n LIMIT 25"

    # 执行查询
    results = graph.run(query).data()

    # 打印查询结果
    for record in results:
        print(record["a.name"])


# 查询与指定节点id为nodeId相连的所有节点，包括节点类别为categoryTypes的节点
def queri_specify_category_nodes(graph):
    # Cypher查询语句
    query = "MATCH (node)-[*]->(related) WHERE id(node) = {nodeId} AND related.category IN {categoryTypes} RETURN DISTINCT related"
    # 执行查询
    results = graph.run(query).data()

    # 打印查询结果
    for record in results:
        print(record["a.name"])


if __name__ == '__main__':
    test(graph)