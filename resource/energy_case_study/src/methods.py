import random
import numpy as np
import networkx as nx

class DriverSelector:
    def __init__(self, G_sub, A_sub, node_list):
        """
        G_sub: NetworkX 子图对象
        A_sub: 子图邻接矩阵 (numpy array)
        node_list: 节点名称列表 (与 A_sub 索引对应)
        """
        self.G = G_sub
        self.A = A_sub
        self.nodes = node_list
        self.num_nodes = len(node_list)

    def select_random(self, k, seed=42):
        """
        [Benchmark 1] 随机选择 K 个节点
        """
        random.seed(seed)
        indices = list(range(self.num_nodes))
        selected_indices = random.sample(indices, k)
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_degree(self, k):
        """
        [Benchmark 2] 基于度中心性 (Hubs) 选择 K 个节点
        """
        # 计算度数 (使用 NetworkX 或直接 sum(A))
        degrees = np.sum(self.A, axis=1)
        
        # 获取索引排序 (从大到小)
        # argsort 默认从小到大，所以取反或切片[::-1]
        sorted_indices = np.argsort(degrees)[::-1]
        
        selected_indices = sorted_indices[:k]
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_hyperdriver_proxy(self, k):
        """
        [HyperDriver Logic] 基于谱能量代理的选择
        
        论文依据: 
        1. 能量代理 E ~ 1 / lambda_max (Section 2.4.2) [cite: 91]
        2. 驱动评分 K_i ~ Delta E_i (Section 2.4.3) [cite: 100]
        
        逻辑:
        我们要找到那些"一旦移除，会导致网络能量 E 发生最大幅度恶化"的节点。
        这意味着这些节点是维持当前网络"低能量可控状态"的关键支撑点(Anchors)。
        因此，控制它们(Input)是最有效的。
        """
        print("Calculating Spectral Energy Proxy scores...")
        
        # 1. 计算原始拉普拉斯矩阵及其最大特征值
        # L = D - A
        D = np.diag(np.sum(self.A, axis=1))
        L_base = D - self.A
        
        # 计算原始谱半径 (Largest Eigenvalue of Laplacian)
        # 注意：对于无向图 L 是半正定，特征值皆为实数 >= 0
        try:
            # 使用 eigvalsh 计算对称矩阵特征值，速度更快且稳定
            evals_base = np.linalg.eigvalsh(L_base)
            lambda_max_base = np.max(evals_base)
        except np.linalg.LinAlgError:
            lambda_max_base = 0.0

        scores = []
        
        # 2. 遍历每个节点，计算"扰动后"的能量变化
        for i in range(self.num_nodes):
            # 模拟移除节点 i:
            # 在矩阵中删除第 i 行和第 i 列
            # 实际上我们构建一个 (N-1)x(N-1) 的矩阵
            A_prime = np.delete(np.delete(self.A, i, axis=0), i, axis=1)
            D_prime_vals = np.sum(A_prime, axis=1)
            L_prime = np.diag(D_prime_vals) - A_prime
            
            try:
                evals_prime = np.linalg.eigvalsh(L_prime)
                lambda_max_prime = np.max(evals_prime)
            except:
                lambda_max_prime = 0.0
            
            # 计算评分:
            # 依据论文，我们关注 Delta E. 
            # E_base ~ 1/lambda_max_base
            # E_prime ~ 1/lambda_max_prime
            # 如果节点重要，移除它会导致系统"更难控制" (Structure degrades), 
            # 通常表现为 lambda_max 下降 (Connectivity/Stiffness drops).
            # 导致 E_prime (1/small) 变得很大。
            # Score = E_prime - E_base
            
            epsilon = 1e-9
            energy_base = 1.0 / (lambda_max_base + epsilon)
            energy_prime = 1.0 / (lambda_max_prime + epsilon)
            
            delta_E = energy_prime - energy_base
            scores.append((i, delta_E))
            
        # 3. 排序: 选择 Delta E 最大的节点 (即移除代价最高的节点)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = [idx for idx, score in scores[:k]]
        return selected_indices, [self.nodes[i] for i in selected_indices]

# 测试代码
if __name__ == "__main__":
    # 简单的 Mock 数据测试
    G_mock = nx.erdos_renyi_graph(20, 0.3, seed=42)
    A_mock = nx.to_numpy_array(G_mock)
    nodes_mock = [str(i) for i in range(20)]
    
    selector = DriverSelector(G_mock, A_mock, nodes_mock)
    
    k = 3
    print("Random:", selector.select_random(k)[0])
    print("Degree:", selector.select_degree(k)[0])
    print("HyperDriver:", selector.select_hyperdriver_proxy(k)[0])