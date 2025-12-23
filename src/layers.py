# src/layers.py
"""
基础层模块 (V4.2 Memory Optimized - Sparse Implementation)
解决 Babu/Gavin 等大数据集 OOM 问题
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# [模块一] 动态边预测器 (Student)
# ============================
class EdgePredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        x_src = x[src_idx]
        x_dst = x[dst_idx]
        cat_feat = torch.cat([x_src, x_dst], dim=-1)
        weights = self.mlp(cat_feat).squeeze(-1)
        return weights


# ============================
# [模块一] 图卷积层 (GCN)
# ============================
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)
        device = x.device
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)

        # Self-loops
        self_loops = torch.arange(N, device=device)
        loop_index = torch.stack([self_loops, self_loops], dim=0)
        loop_weight = torch.ones(N, device=device, dtype=x.dtype)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=device, dtype=x.dtype)
        deg.index_add_(0, src, edge_weight)
        deg = deg.clamp(min=1e-6)
        
        norm = edge_weight / (deg[src] * deg[dst]).sqrt()
        
        msg = x[src] * norm.unsqueeze(-1)
        out = torch.zeros(N, self.lin.in_features, device=device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        out = self.lin(out)
        return out


# ============================
# [模块二] 动态超图构建器 (Sparse Optimized)
# ============================
class DynamicHypergraphBuilder(nn.Module):
    """
    V4.2 优化：返回稀疏索引而不是稠密矩阵，解决 OOM。
    """
    def __init__(self, k_neighbors: int = 10):
        super().__init__()
        self.k = k_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回: H_sparse (torch.sparse_coo_tensor) [N, N]
        """
        N = x.size(0)
        device = x.device

        # 1. 计算距离 (使用 cdist 更省内存)
        # 即使这里产生了 N*N，但在 TopK 后会被释放，不会像 H 那样一直存着
        dist_mat = torch.cdist(x, x, p=2) 
        
        # 2. Top-K
        # indices: [N, k+1]
        _, indices = torch.topk(dist_mat, k=self.k + 1, dim=1, largest=False)
        
        # 3. 构建稀疏索引 (Sparse Indices)
        # 行: 节点 ID (被包含的节点)
        # 列: 超边 ID (中心节点)
        row_idx = indices.reshape(-1) # [N * (k+1)]
        col_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, self.k + 1).reshape(-1)
        
        indices_tensor = torch.stack([row_idx, col_idx], dim=0)
        values_tensor = torch.ones(indices_tensor.size(1), device=device, dtype=torch.float32)
        
        # 创建稀疏张量
        H_sparse = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N, N))
        
        return H_sparse


# ============================
# [模块二] 超图卷积层 (Sparse Optimized)
# ============================
class HypergraphConvLayer(nn.Module):
    """
    支持稀疏矩阵输入的 HGNN。
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, H_sparse: torch.Tensor) -> torch.Tensor:
        """
        x: [N, F]
        H_sparse: [N, N] Sparse Tensor
        """
        N = x.size(0)
        device = x.device
        
        # H 是稀疏的，我们不能直接 sum(dim=1) 得到 dense degree
        # 需要手动计算度
        # H indices: [2, E_total] where row=node, col=hyperedge
        H_indices = H_sparse._indices()
        
        # 1. 计算节点度 Dv
        # Dv[i] = sum of H[i, :]
        Dv = torch.zeros(N, device=device, dtype=x.dtype)
        Dv.index_add_(0, H_indices[0], torch.ones(H_indices.shape[1], device=device))
        Dv = Dv.clamp(min=1e-6)
        Dv_inv_sqrt = torch.pow(Dv, -0.5).view(-1, 1)

        # 2. 计算超边度 De
        # De[j] = sum of H[:, j]
        # 因为我们是 k-NN 建图，每个超边固定连接 k+1 个节点
        # 所以 De 其实是常数 k+1，但为了通用性还是算一下
        De = torch.zeros(N, device=device, dtype=x.dtype)
        De.index_add_(0, H_indices[1], torch.ones(H_indices.shape[1], device=device))
        De = De.clamp(min=1e-6)
        De_inv = torch.pow(De, -1.0).view(-1, 1)

        # HGNN Propagation:
        # X -> Dv^-0.5 X
        x_norm = x * Dv_inv_sqrt
        
        # -> H^T (Dv^-0.5 X)
        # 稀疏矩阵乘法: (N, N)^T @ (N, F) = (N, N) @ (N, F) -> (N, F)
        # torch.sparse.mm 支持 (Sparse, Dense) -> Dense
        x_e = torch.sparse.mm(H_sparse.t(), x_norm)
        
        # -> De^-1 (...)
        x_e = x_e * De_inv
        
        # -> H (...)
        x_v = torch.sparse.mm(H_sparse, x_e)
        
        # -> Dv^-0.5 (...)
        x_v = x_v * Dv_inv_sqrt
        
        # Linear
        out = self.lin(x_v)
        return out


# ============================
# 时间聚合 & 门控 (通用)
# ============================
class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x_seq)
        return h_n[-1]

class NodeGatingLayer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )

    def forward(self, z_graph: torch.Tensor, z_hyper: torch.Tensor):
        z_cat = torch.cat([z_graph, z_hyper], dim=-1)
        g = torch.sigmoid(self.gate_mlp(z_cat))
        z_mix = g * z_graph + (1.0 - g) * z_hyper
        return g, z_mix