#!/usr/bin/env python
# coding: utf-8
"""
生成一体化数据库架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'PingFang SC', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# 定义颜色方案
color_product = '#4A90E2'  # 蓝色 - 产品层
color_engine = '#50C878'   # 绿色 - 引擎层
color_arch = '#FF6B6B'     # 红色 - 架构层
color_text = '#2C3E50'     # 深灰色 - 文字

# 绘制标题
title = ax.text(5, 13.5, '一体化数据库架构', 
                ha='center', va='center', 
                fontsize=24, fontweight='bold', 
                color=color_text)

# ========== 第一层：一体化产品 ==========
y_product = 11
product_box = FancyBboxPatch((0.5, y_product-0.8), 9, 1.6,
                            boxstyle="round,pad=0.1", 
                            edgecolor=color_product, 
                            facecolor=color_product, 
                            alpha=0.2, 
                            linewidth=2.5,
                            zorder=1)
ax.add_patch(product_box)

product_title = ax.text(5, y_product+0.3, '一体化产品', 
                       ha='center', va='center',
                       fontsize=18, fontweight='bold',
                       color=color_product)

# 产品层的三个子项
product_items = [
    ('多工作负载', 'TP & AP 融合\n行列存储一体化', 2.5),
    ('多模', 'SQL + NoSQL\nTable、KV、HBase API', 5),
    ('多兼容模式', '兼容 MySQL、Oracle', 7.5)
]

for label, desc, x_pos in product_items:
    # 子项框
    item_box = FancyBboxPatch((x_pos-0.8, y_product-0.5), 1.6, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor=color_product,
                              facecolor='white',
                              linewidth=2,
                              zorder=2)
    ax.add_patch(item_box)
    
    # 标签
    ax.text(x_pos, y_product+0.15, label,
           ha='center', va='center',
           fontsize=14, fontweight='bold',
           color=color_product)
    
    # 描述
    ax.text(x_pos, y_product-0.25, desc,
           ha='center', va='center',
           fontsize=10,
           color=color_text)

# ========== 第二层：一体化引擎 ==========
y_engine = 8
engine_box = FancyBboxPatch((0.5, y_engine-0.8), 9, 1.6,
                           boxstyle="round,pad=0.1",
                           edgecolor=color_engine,
                           facecolor=color_engine,
                           alpha=0.2,
                           linewidth=2.5,
                           zorder=1)
ax.add_patch(engine_box)

engine_title = ax.text(5, y_engine+0.3, '一体化引擎',
                      ha='center', va='center',
                      fontsize=18, fontweight='bold',
                      color=color_engine)

# 引擎层的三个子项
engine_items = [
    ('一体化存储', '行存、列存\n行列混存', 2.5),
    ('一体化 SQL', '向量化执行引擎 2.0\n混合负载优化执行层', 5),
    ('一体化 KV', 'OBKV 引擎\n可扩展性组件', 7.5)
]

for label, desc, x_pos in engine_items:
    # 子项框
    item_box = FancyBboxPatch((x_pos-0.8, y_engine-0.5), 1.6, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor=color_engine,
                              facecolor='white',
                              linewidth=2,
                              zorder=2)
    ax.add_patch(item_box)
    
    # 标签
    ax.text(x_pos, y_engine+0.15, label,
           ha='center', va='center',
           fontsize=14, fontweight='bold',
           color=color_engine)
    
    # 描述
    ax.text(x_pos, y_engine-0.25, desc,
           ha='center', va='center',
           fontsize=10,
           color=color_text)

# ========== 第三层：一体化架构 ==========
y_arch = 5
arch_box = FancyBboxPatch((0.5, y_arch-0.8), 9, 1.6,
                         boxstyle="round,pad=0.1",
                         edgecolor=color_arch,
                         facecolor=color_arch,
                         alpha=0.2,
                         linewidth=2.5,
                         zorder=1)
ax.add_patch(arch_box)

arch_title = ax.text(5, y_arch+0.3, '一体化架构',
                    ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color=color_arch)

# 架构层的两个子项
arch_items = [
    ('单机分布式一体化', '灵活应对从小规模\n到大规模的场景需求', 3.5),
    ('存算一体 & 存算分离融合', 'Share Nothing → Share Storage\n+ Share Nothing\n自动冷热数据分离', 6.5)
]

for label, desc, x_pos in arch_items:
    # 子项框
    item_box = FancyBboxPatch((x_pos-1.1, y_arch-0.5), 2.2, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor=color_arch,
                              facecolor='white',
                              linewidth=2,
                              zorder=2)
    ax.add_patch(item_box)
    
    # 标签
    ax.text(x_pos, y_arch+0.15, label,
           ha='center', va='center',
           fontsize=13, fontweight='bold',
           color=color_arch)
    
    # 描述
    ax.text(x_pos, y_arch-0.25, desc,
           ha='center', va='center',
           fontsize=9,
           color=color_text)

# ========== 绘制连接线 ==========
# 从产品层到引擎层的连接
for x_pos in [2.5, 5, 7.5]:
    arrow = FancyArrowPatch((x_pos, y_product-0.8), (x_pos, y_engine+0.8),
                           arrowstyle='->', 
                           mutation_scale=20,
                           color='#95A5A6',
                           linewidth=1.5,
                           alpha=0.6,
                           zorder=0)
    ax.add_patch(arrow)

# 从引擎层到架构层的连接
for x_pos in [3.5, 6.5]:
    arrow = FancyArrowPatch((x_pos, y_engine-0.8), (x_pos, y_arch+0.8),
                           arrowstyle='->',
                           mutation_scale=20,
                           color='#95A5A6',
                           linewidth=1.5,
                           alpha=0.6,
                           zorder=0)
    ax.add_patch(arrow)

# 保存图片
output_path = '一体化数据库架构图.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"架构图已保存到: {output_path}")
plt.close()

