# src/layers.py
"""
Base layer module (V4.2 Memory Optimized - Sparse Implementation)
Solving OOM problems on large datasets such as Babu/Gavin
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# [Module 1] Dynamic Edge Predictor (Student)
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
# [Module 1] Graph Convolutional Layers (GCN)
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
# [Module 2] Dynamic Hypergraph Builder (Sparse Optimized)
# ============================
class DynamicHypergraphBuilder(nn.Module):
    """
    Optimization: Return sparse indices instead of dense matrices to resolve OOM (Out of Memory) errors.
    """
    def __init__(self, k_neighbors: int = 10):
        super().__init__()
        self.k = k_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: H_sparse (torch.sparse_coo_tensor) [N, N]
        """
        N = x.size(0)
        device = x.device

        # 1. Calculate distance (using cdist saves memory)
        # Even if N*N is generated here, it will be released after TopK, unlike H which will be stored indefinitely.
        dist_mat = torch.cdist(x, x, p=2) 
        
        # 2. Top-K
        # indices: [N, k+1]
        _, indices = torch.topk(dist_mat, k=self.k + 1, dim=1, largest=False)
        
        # 3. Constructing Sparse Indices
        # Line: Node ID (the included node)
        # Column: Hyperedge ID (Center Node)
        row_idx = indices.reshape(-1) # [N * (k+1)]
        col_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, self.k + 1).reshape(-1)
        
        indices_tensor = torch.stack([row_idx, col_idx], dim=0)
        values_tensor = torch.ones(indices_tensor.size(1), device=device, dtype=torch.float32)
        
        # Creating sparse tensors
        H_sparse = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N, N))
        
        return H_sparse


# ============================
# [Module 2] Hypergraph Convolutional Layer (Sparse Optimized)
# ============================
class HypergraphConvLayer(nn.Module):
    """
    HGNN that supports sparse matrix input.
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
        
        # H is sparse, so we cannot directly obtain the dense degree by sum(dim=1).
        # Degree needs to be calculated manually
        # H indices: [2, E_total] where row=node, col=hyperedge
        H_indices = H_sparse._indices()
        
        # 1. Calculate the node degree Dv
        # Dv[i] = sum of H[i, :]
        Dv = torch.zeros(N, device=device, dtype=x.dtype)
        Dv.index_add_(0, H_indices[0], torch.ones(H_indices.shape[1], device=device))
        Dv = Dv.clamp(min=1e-6)
        Dv_inv_sqrt = torch.pow(Dv, -0.5).view(-1, 1)

        # 2. Calculate the hypermargin De.
        # De[j] = sum of H[:, j]
        # Because we are using k-NN to construct the graph, each hyperedge connects k+1 nodes.
        # Therefore, De is actually a constant k+1, but for the sake of generality, let's calculate it again.
        De = torch.zeros(N, device=device, dtype=x.dtype)
        De.index_add_(0, H_indices[1], torch.ones(H_indices.shape[1], device=device))
        De = De.clamp(min=1e-6)
        De_inv = torch.pow(De, -1.0).view(-1, 1)

        # HGNN Propagation:
        # X -> Dv^-0.5 X
        x_norm = x * Dv_inv_sqrt
        
        # -> H^T (Dv^-0.5 X)
        # Sparse matrix multiplication: (N, N)^T @ (N, F) = (N, N) @ (N, F) -> (N, F)
        # torch.sparse.mm supports (Sparse, Dense) -> Dense
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
# Time aggregation & gating (general)
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