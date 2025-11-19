#!/usr/bin/env python
# coding: utf-8
"""
使用 diagrams 库生成一体化数据库架构图
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

# 设置图表属性
graph_attr = {
    "fontsize": "16",
    "fontname": "Arial Unicode MS, SimHei, STHeiti, PingFang SC, Microsoft YaHei",
    "bgcolor": "white",
    "rankdir": "TB",  # 从上到下
    "splines": "ortho",  # 使用正交线条
    "nodesep": "1.2",
    "ranksep": "2.0",
    "pad": "0.5"
}

node_attr = {
    "fontname": "Arial Unicode MS, SimHei, STHeiti, PingFang SC, Microsoft YaHei",
    "fontsize": "11",
    "shape": "box",
    "style": "rounded,filled",
    "width": "2.5",
    "height": "1.2"
}

edge_attr = {
    "color": "#666666",
    "style": "solid",
    "penwidth": "2.5"
}

with Diagram(
    "一体化数据库架构",
    filename="一体化数据库架构图_diagrams",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    outformat="png"
):
    # ========== 第一层：一体化产品 ==========
    with Cluster(
        "一体化产品",
        graph_attr={
            "bgcolor": "#E3F2FD",
            "style": "rounded,filled",
            "color": "#1976D2",
            "penwidth": "3",
            "fontsize": "18",
            "fontname": "Arial Unicode MS, SimHei, STHeiti, PingFang SC, Microsoft YaHei"
        }
    ):
        # 多工作负载
        workload = Custom(
            "多工作负载\nTP & AP 融合\n行列存储一体化",
            ""
        )
        
        # 多模
        multimodal = Custom(
            "多模\nSQL + NoSQL\nTable、KV、HBase API",
            ""
        )
        
        # 多兼容模式
        compatibility = Custom(
            "多兼容模式\n兼容 MySQL、Oracle",
            ""
        )

    # ========== 第二层：一体化引擎 ==========
    with Cluster(
        "一体化引擎",
        graph_attr={
            "bgcolor": "#E8F5E9",
            "style": "rounded,filled",
            "color": "#388E3C",
            "penwidth": "3",
            "fontsize": "18",
            "fontname": "Arial Unicode MS, SimHei, STHeiti, PingFang SC, Microsoft YaHei"
        }
    ):
        # 一体化存储
        storage = Custom(
            "一体化存储\n行存、列存\n行列混存",
            ""
        )
        
        # 一体化 SQL
        sql_engine = Custom(
            "一体化 SQL\n向量化执行引擎 2.0\n混合负载优化执行层",
            ""
        )
        
        # 一体化 KV
        kv_engine = Custom(
            "一体化 KV\nOBKV 引擎\n可扩展性组件",
            ""
        )

    # ========== 第三层：一体化架构 ==========
    with Cluster(
        "一体化架构",
        graph_attr={
            "bgcolor": "#FFEBEE",
            "style": "rounded,filled",
            "color": "#D32F2F",
            "penwidth": "3",
            "fontsize": "18",
            "fontname": "Arial Unicode MS, SimHei, STHeiti, PingFang SC, Microsoft YaHei"
        }
    ):
        # 单机分布式一体化
        architecture = Custom(
            "单机分布式一体化\n灵活应对从小规模\n到大规模的场景需求",
            ""
        )
        
        # 存算一体 & 存算分离融合
        storage_compute = Custom(
            "存算一体 & 存算分离融合\nShare Nothing → Share Storage\n+ Share Nothing\n自动冷热数据分离",
            ""
        )

    # ========== 连接关系 ==========
    # 产品层到引擎层
    workload >> Edge(color="#1976D2", style="bold", penwidth="3") >> storage
    multimodal >> Edge(color="#1976D2", style="bold", penwidth="3") >> sql_engine
    compatibility >> Edge(color="#1976D2", style="bold", penwidth="3") >> kv_engine
    
    # 引擎层到架构层
    storage >> Edge(color="#388E3C", style="bold", penwidth="3") >> architecture
    sql_engine >> Edge(color="#388E3C", style="bold", penwidth="3") >> storage_compute
    kv_engine >> Edge(color="#388E3C", style="bold", penwidth="3") >> storage_compute

print("架构图已生成: 一体化数据库架构图_diagrams.png")
