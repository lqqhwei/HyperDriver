import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

class SubgraphSelector:
    def __init__(self, G, target_size=(50, 150)):
        """
        G: NetworkX graph (Static PPIN)
        target_size: tuple (min_nodes, max_nodes) 期望的子图规模
        """
        self.G = G
        self.min_size = target_size[0]
        self.max_size = target_size[1]

    def get_best_subgraph(self):
        """
        尝试寻找最适合做 Control Energy 实验的子图。
        优先策略：高密度的社区 (Community)。
        备选策略：以 Hub 为中心的局部邻域 (Hub-BFS)。
        """
        print(f"Searching for optimal subgraph (Target size: {self.min_size}-{self.max_size})...")
        
        # 策略 1: 尝试社区发现
        best_community = self._find_best_community()
        if best_community:
            print(f"✅ Method: Community Detection. Selected cluster size: {len(best_community)}")
            return self._extract_subgraph_from_nodes(best_community)
        
        # 策略 2: 如果找不到合适大小的社区，使用 Hub 扩展
        print("⚠️ No perfect community found. Switching to Hub-Expansion strategy.")
        best_hub_region = self._grow_from_hub()
        print(f"✅ Method: Hub Expansion. Selected region size: {len(best_hub_region)}")
        return self._extract_subgraph_from_nodes(best_hub_region)

    def _find_best_community(self):
        """
        使用贪婪模块度最大化算法寻找社区
        """
        try:
            # 寻找社区 (这是一个由 node sets 组成的 list)
            communities = list(greedy_modularity_communities(self.G))
            
            candidates = []
            for c in communities:
                if self.min_size <= len(c) <= self.max_size:
                    candidates.append(list(c))
            
            if not candidates:
                return None
            
            # 在候选者中，选择密度(Density)最大的社区
            # 密度高意味着控制关系更复杂，更能体现最小能量算法的优势
            best_c = None
            max_density = -1.0
            
            for c_nodes in candidates:
                subg = self.G.subgraph(c_nodes)
                density = nx.density(subg)
                if density > max_density:
                    max_density = density
                    best_c = c_nodes
            
            return best_c

        except Exception as e:
            print(f"Community detection failed: {e}")
            return None

    def _grow_from_hub(self):
        """
        从度最大的节点开始 BFS 扩散，直到达到 max_size
        """
        # 找到度最大的节点
        degree_dict = dict(self.G.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
        top_hub = sorted_nodes[0][0]
        
        selected_nodes = {top_hub}
        queue = [top_hub]
        
        while len(selected_nodes) < self.max_size and queue:
            current = queue.pop(0)
            neighbors = list(self.G.neighbors(current))
            
            # 按邻居的度数排序，优先纳入重要的邻居
            neighbors.sort(key=lambda n: degree_dict.get(n, 0), reverse=True)
            
            for n in neighbors:
                if n not in selected_nodes:
                    selected_nodes.add(n)
                    queue.append(n)
                    if len(selected_nodes) >= self.max_size:
                        break
        
        return list(selected_nodes)

    def _extract_subgraph_from_nodes(self, nodes):
        """
        根据节点列表提取子图对象、邻接矩阵和映射
        """
        subgraph = self.G.subgraph(nodes).copy()
        
        # 重新排序节点以保证矩阵索引对齐
        sorted_nodes = sorted(list(subgraph.nodes()))
        
        # 提取邻接矩阵 (numpy array)
        adj_matrix = nx.to_numpy_array(subgraph, nodelist=sorted_nodes)
        
        # 建立局部索引映射: Subgraph Index -> Protein Name
        idx_map = {i: name for i, name in enumerate(sorted_nodes)}
        
        # 返回: NetworkX对象, 邻接矩阵, 节点列表, 索引映射
        return subgraph, adj_matrix, sorted_nodes, idx_map
