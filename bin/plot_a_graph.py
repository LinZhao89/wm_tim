import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.patches import Polygon

# 设置随机种子以保证结果可复现
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. 定义节点 (Test Items)
# ==========================================
voltage_nodes = [f'V_th_{i}' for i in range(1, 6)] + [f'V_bd_{i}' for i in range(1, 4)] + ['V_g', 'V_d', 'V_s']
res_nodes = [f'R_on_{i}' for i in range(1, 6)] + [f'R_cont_{i}' for i in range(1, 4)] + ['R_sheet', 'R_metal']
current_nodes = [f'I_leak_{i}' for i in range(1, 6)] + ['I_ds_sat', 'I_off', 'I_sub']
other_nodes = [f'Cap_ox_{i}' for i in range(1, 4)] + ['Temp_die', 'Stress_X', 'Stress_Y', 'Bin_Code']

all_nodes = voltage_nodes + res_nodes + current_nodes + other_nodes

# 创建图
G = nx.Graph()
G.add_nodes_from(all_nodes)

# ==========================================
# 2. 构建连接和结构
# ==========================================

# A. 随机骨干连接
for _ in range(35):
    u, v = random.sample(all_nodes, 2)
    G.add_edge(u, v)

# B. 定义三角形结构 (Triangles)
triangles = [
    (voltage_nodes[0], res_nodes[0], current_nodes[0]), # V=IR 关系
    (voltage_nodes[1], voltage_nodes[2], other_nodes[0]),
    (res_nodes[1], res_nodes[2], current_nodes[1])
]
for tri in triangles:
    nx.add_cycle(G, tri)

# C. 定义长方形结构 (Rectangles)
rectangles = [
    (voltage_nodes[3], res_nodes[3], voltage_nodes[4], res_nodes[4]),
    (current_nodes[2], other_nodes[1], current_nodes[3], other_nodes[2])
]
for rect in rectangles:
    nx.add_cycle(G, rect)

# ==========================================
# 3. 可视化设置
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8))

# 布局算法
pos = nx.spring_layout(G, k=0.25, iterations=50, seed=10)

# --- 新增功能：绘制结构背景阴影 ---
# 绘制三角形阴影 (淡黄色)
for tri_nodes in triangles:
    coords = [pos[n] for n in tri_nodes]
    poly = Polygon(coords, closed=True, facecolor='#FFEB3B', alpha=0.3, edgecolor='none', zorder=0)
    ax.add_patch(poly)

# 绘制长方形阴影 (淡紫色)
for rect_nodes in rectangles:
    # 需要按顺时针或逆时针顺序获取坐标，否则形状会交叉
    # 对于 4-cycle，可以先找出子图然后按顺序提取
    subgraph = G.subgraph(rect_nodes)
    cycle_nodes = list(nx.find_cycle(subgraph))
    ordered_nodes = [edge[0] for edge in cycle_nodes]
    coords = [pos[n] for n in ordered_nodes]
    
    poly = Polygon(coords, closed=True, facecolor='#CE93D8', alpha=0.3, edgecolor='none', zorder=0)
    ax.add_patch(poly)

# --- 绘制节点和边 ---
# 定义颜色映射
color_map = []
for node in G.nodes():
    if node in voltage_nodes:
        color_map.append('#A0CBE8') # Voltage
    elif node in res_nodes:
        color_map.append('#FFBE7D') # Resistance
    elif node in current_nodes:
        color_map.append('#8CD17D') # Current
    else:
        color_map.append('#D3D3D3') # Others

# 绘制边
nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.6, edge_color='#555555', ax=ax)
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=600, node_color=color_map, edgecolors='grey', ax=ax)
# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold', ax=ax)

# 去除坐标轴和边框
ax.axis('off')
plt.title("Homogeneous Graph with Structural Highlights\n(Yellow=Triangles, Purple=Rectangles)", fontsize=14)

# ==========================================
# 4. 保存结果
# ==========================================
plt.tight_layout()
plt.savefig("test_items_graph_highlighted.svg", format="svg")
plt.show()