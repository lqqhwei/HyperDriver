import numpy as np
from scipy import linalg

class ControlEnergyCalculator:
    def __init__(self, adj_matrix):
        """
        初始化能量计算器
        adj_matrix: 子图的邻接矩阵 (NxN numpy array)
        """
        self.A_adj = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        
        # 1. 构建系统矩阵 A_sys
        # 动力学方程: dx/dt = -L x (一致性协议/扩散过程)
        # L = D - A
        degrees = np.sum(adj_matrix, axis=1)
        self.L = np.diag(degrees) - adj_matrix
        
        # 2. 稳定性处理 (Stability Perturbation)
        # 为了保证 Lyapunov 方程有唯一解，矩阵 A 必须是 Hurwitz 稳定的 (特征值实部 < 0)。
        # 原始 -L 的特征值 <= 0，且包含 0。
        # 我们引入一个微小的衰减率 epsilon (模拟生物降解)，使系统渐进稳定。
        self.epsilon = 0.5  # 对于 PPI 网络，0.1~1.0 都是合理的衰减系数
        self.A_sys = -self.L - self.epsilon * np.eye(self.num_nodes)

    def compute_energy(self, driver_indices):
        """
        计算给定驱动节点集合的"平均控制能量"。
        driver_indices: list of int, 选中的驱动节点索引
        
        Returns:
            energy_score: float (数值越小越好)
        """
        k = len(driver_indices)
        if k == 0:
            return float('inf')

        # 1. 构建输入矩阵 B (NxK)
        # 只有在 driver_indices 对应的行是 1，其余是 0
        B = np.zeros((self.num_nodes, k))
        for col_idx, node_idx in enumerate(driver_indices):
            B[node_idx, col_idx] = 1.0

        # 2. 求解连续 Lyapunov 方程 (Continuous Lyapunov Equation)
        # A X + X A^T = Q
        # 对应我们的形式: A_sys Wc + Wc A_sys^T + B B^T = 0
        # 所以 Q = -B B^T
        Q = -np.dot(B, B.T)
        
        try:
            # scipy.linalg.solve_continuous_lyapunov(a, q) 求解 AX + XA^H = Q
            Wc = linalg.solve_continuous_lyapunov(self.A_sys, Q)
        except Exception as e:
            print(f"Lyapunov Solver failed: {e}")
            return float('inf')

        # 3. 计算能量指标
        # 理论: Minimum Energy ~ Trace(Inv(Wc))
        # Gramian Wc 的特征值度量了系统在各个方向上的可控性。
        # 特征值越大 -> 可控性越好 -> 需要的能量越少。
        # 能量 E 与 Wc 的特征值成反比。
        
        try:
            # 计算 Wc 的特征值
            evals = linalg.eigvalsh(Wc)
            
            # 过滤极小的特征值以避免除以零 (数值截断)
            # 实际计算中 Wc 应该是正定的，但浮点误差可能产生微小负数或0
            min_tol = 1e-12
            evals = evals[evals > min_tol]
            
            if len(evals) == 0:
                return float('inf')
            
            # 指标: Average Energy = Trace(Wc^-1) = Sum(1/lambda_i)
            energy = np.sum(1.0 / evals)
            
            return energy

        except np.linalg.LinAlgError:
            return float('inf')

# 测试代码
if __name__ == "__main__":
    # 创建一个小测试图
    import networkx as nx
    G_test = nx.path_graph(10)
    A_test = nx.to_numpy_array(G_test)
    
    calculator = ControlEnergyCalculator(A_test)
    
    # 比较两种驱动方案
    # 方案 A: 选端点 (通常较难控制全图)
    drivers_A = [0] 
    energy_A = calculator.compute_energy(drivers_A)
    
    # 方案 B: 选中心点 (通常容易控制)
    drivers_B = [4]
    energy_B = calculator.compute_energy(drivers_B)
    
    print(f"Energy (End Node): {energy_A:.4e}")
    print(f"Energy (Center Node): {energy_B:.4e}")
    
    if energy_B < energy_A:
        print("✅ 测试通过: 中心节点控制能量更低。")
    else:
        print("❌ 测试失败: 结果不符合直觉。")