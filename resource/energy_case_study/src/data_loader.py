import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix

class DataLoader:
    def __init__(self, static_dir,label_dir):
        self.static_dir = static_dir
        self.label_dir = label_dir
        self.static_file = os.path.join(static_dir, 'Static_PPIN.txt')
        self.label_file = os.path.join(label_dir, 'Node_Labels_with_essential.csv')
        
        # 节点映射表：Protein Name -> Index
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.num_nodes = 0
        
    def load_static_graph(self):
        """
        加载静态PPI网络，返回 NetworkX 对象和邻接矩阵
        """
        print(f"Loading Static PPIN from {self.static_file}...")
        
        # 1. 读取边列表，建立节点集合
        edges = []
        nodes = set()
        
        with open(self.static_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                u, v = parts[0], parts[1]
                # 兼容可能的权重列，如果没有默认1.0
                w = float(parts[2]) if len(parts) > 2 else 1.0
                
                edges.append((u, v, w))
                nodes.add(u)
                nodes.add(v)
        
        # 2. 建立索引映射
        sorted_nodes = sorted(list(nodes))
        self.node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(sorted_nodes)}
        self.num_nodes = len(sorted_nodes)
        
        print(f"Graph loaded. Nodes: {self.num_nodes}, Edges: {len(edges)}")
        
        # 3. 构建 NetworkX 图
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
            
        return G

    def load_essential_labels(self):
        """
        加载必需蛋白标签，返回一个 binary list (对应 idx 顺序)
        """
        print(f"Loading Labels from {self.label_file}...")
        df = pd.read_csv(self.label_file)
        
        # 假设 CSV 列名为: 'Protein', 'Essential' (或其他，需根据实际文件调整)
        # 这里做一个简单的容错处理，打印列名以便调试
        # print("Columns:", df.columns)
        
        # 创建必需蛋白集合
        # 过滤 Essential == 'E' (假设 E 代表 Essential, 根据实际数据调整)
        # 这里通常 Essential 列是 'E' (Essential) 或 'NE' (Non-Essential)
        essential_nodes = set(df[df['Essential'] == 'E']['Protein'].values)
        
        labels = np.zeros(self.num_nodes, dtype=int)
        count = 0
        for i in range(self.num_nodes):
            node_name = self.idx_to_node[i]
            if node_name in essential_nodes:
                labels[i] = 1
                count += 1
                
        print(f"Essential proteins mapped: {count}/{self.num_nodes}")
        return labels

    def get_subgraph_data(self, G, node_indices):
        """
        辅助函数：根据节点索引列表提取子图的数据结构
        用于 Case Study
        """
        subgraph_nodes = [self.idx_to_node[i] for i in node_indices]
        G_sub = G.subgraph(subgraph_nodes).copy()
        
        # 重新映射子图的索引从 0 到 M
        sub_nodes_list = sorted(list(G_sub.nodes()))
        sub_node_to_idx = {n: i for i, n in enumerate(sub_nodes_list)}
        
        # 提取子图邻接矩阵 A_sub
        A_sub = nx.to_numpy_array(G_sub, nodelist=sub_nodes_list)
        
        return G_sub, A_sub, sub_node_to_idx, sub_nodes_list

# 测试代码 (运行时可注释掉)
if __name__ == "__main__":
    # 假设你的数据在 current_dir/data
    loader = DataLoader("./data")
    G = loader.load_static_graph()
    labels = loader.load_essential_labels()