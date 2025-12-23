# src/hyper_driver.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    GraphConvLayer, 
    HypergraphConvLayer, 
    TemporalEncoder, 
    NodeGatingLayer, 
    EdgePredictor, 
    DynamicHypergraphBuilder
)


@dataclass
class HyperDriverConfig:
    """
    HyperDriver 模型配置 (V4.2 Optimized: Sparse + Residual)
    """
    in_feats: int                  
    hidden_dim: int = 64           
    num_graph_layers: int = 2      
    num_hyper_layers: int = 1      
    num_time_layers: int = 1       
    num_classes: int = 1           
    # [优化] 只保留精细尺度，减少计算量和显存
    num_scales: List[int] = field(default_factory=lambda: [5, 10, 15]) 
    # [优化] 提高 Dropout
    dropout: float = 0.5


class HyperDriver(nn.Module):
    """
    HyperDriver V4.2 主模型
    """

    def __init__(self, config: HyperDriverConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # -------- 特征投影 --------
        self.input_proj = nn.Linear(config.in_feats, H)

        # -------- 模块一：动态图学习 (Student) --------
        self.edge_predictor = EdgePredictor(in_channels=H, hidden_dim=H // 2)

        self.graph_convs = nn.ModuleList()
        for i in range(config.num_graph_layers):
            self.graph_convs.append(GraphConvLayer(H, H))

        self.graph_temporal_encoder = TemporalEncoder(H, H, config.num_time_layers)

        # -------- 模块二：多尺度动态超图 (Sparse) --------
        self.hyper_builders = nn.ModuleList()
        self.hyper_convs_per_scale = nn.ModuleList()

        for k in config.num_scales:
            self.hyper_builders.append(DynamicHypergraphBuilder(k_neighbors=k))
            scale_layers = nn.ModuleList()
            for j in range(config.num_hyper_layers):
                scale_layers.append(HypergraphConvLayer(H, H))
            self.hyper_convs_per_scale.append(scale_layers)

        self.hyper_temporal_encoder = TemporalEncoder(H, H, config.num_time_layers)

        # -------- 模块三：门控与分类 (Residual) --------
        self.gating = NodeGatingLayer(in_channels=H)
        self.hyper_residual_proj = nn.Linear(H, H) # 用于残差对齐

        # 这里的 alpha 作为一个可学习参数，控制 residual 的初始比例
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.classifier = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, config.num_classes),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x_seq: torch.Tensor,
        static_edge_index: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        标准的训练前向传播
        """
        T, N, F_in = x_seq.shape
        device = x_seq.device
        static_edge_index = static_edge_index.to(device)

        # 1. 特征投影
        x_emb = self.input_proj(x_seq)

        graph_feats_list = []
        hyper_feats_list = []
        predicted_weights_list = []

        # 2. 按时间步循环
        for t in range(T):
            x_t = x_emb[t]

            # ========== 模块一：动态图分支 ==========
            w_pred = self.edge_predictor(x_t, static_edge_index)
            predicted_weights_list.append(w_pred)

            h_graph = x_t
            for conv in self.graph_convs:
                h_graph = conv(h_graph, edge_index=static_edge_index, edge_weight=w_pred)
                h_graph = F.relu(h_graph)
                h_graph = self.dropout(h_graph)
            graph_feats_list.append(h_graph)

            # ========== 模块二：动态超图分支 (Sparse) ==========
            if len(self.config.num_scales) > 0:
                scale_outputs = []
                for idx, k in enumerate(self.config.num_scales):
                    # Builder 返回的是稀疏张量 H_sparse
                    builder = self.hyper_builders[idx]
                    H_sparse = builder(x_t)

                    h_hyper_scale = x_t
                    scale_convs = self.hyper_convs_per_scale[idx]
                    for conv in scale_convs:
                        h_hyper_scale = conv(h_hyper_scale, H_sparse)
                        h_hyper_scale = F.relu(h_hyper_scale)
                        h_hyper_scale = self.dropout(h_hyper_scale)
                    scale_outputs.append(h_hyper_scale)
                
                h_hyper_t = torch.stack(scale_outputs, dim=0).mean(dim=0)
            else:
                h_hyper_t = torch.zeros_like(x_t)
            hyper_feats_list.append(h_hyper_t)

        # 3. 时序聚合
        # [T, N, H] -> [N, T, H]
        graph_seq_tensor = torch.stack(graph_feats_list, dim=0).permute(1, 0, 2)
        z_graph = self.graph_temporal_encoder(graph_seq_tensor)

        hyper_seq_tensor = torch.stack(hyper_feats_list, dim=0).permute(1, 0, 2)
        z_hyper = self.hyper_temporal_encoder(hyper_seq_tensor)

        # 4. 模块三：残差融合
        # g 在这里用于 gating mask 的生成 (N, 1)
        g, _ = self.gating(z_graph, z_hyper) 
        
        z_hyper_proj = self.hyper_residual_proj(z_hyper)
        # 显式使用 sigmoid(alpha) 控制混合比例，确保训练初期稳定
        z_mix = z_graph + torch.sigmoid(self.alpha) * 0.1 * z_hyper_proj

        # 5. 分类头
        logits = self.classifier(z_mix)

        return {
            "logits": logits,
            "z_graph": z_graph,
            "z_hyper": z_hyper,
            "g": g,
            "z_mix": z_mix,
            "predicted_weights_list": predicted_weights_list,
        }

    @torch.no_grad()
    def get_consensus_structure(
        self,
        x_seq: torch.Tensor,
        static_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [NEW] 获取用于 Average Energy Controllability 分析的"时间平均混合结构"。
        
        返回:
            avg_weights: (E_static, ) 静态骨架上的动态权重平均值 (mean of w_pred)
            avg_hyper_adj: (N, N)     超图分支的有效邻接矩阵平均值 (mean of A_hyper)
            g: (N, )                  节点级门控值 (time-invariant structure preference)
        
        逻辑:
        1. 遍历时间步，累积 w_pred。
        2. 遍历时间步，构建 H_sparse，计算 L_hyper 对应的归一化邻接矩阵 A = Dv^-0.5 H De^-1 H^T Dv^-0.5。
           虽然 H 是稀疏的，但为了后续谱分解方便，我们在这里累积并返回 dense 格式的 N x N 矩阵。
           (对于 N <= 5000 的数据集，5000^2 * 4B ≈ 100MB，完全可接受)
        3. 返回 g。
        """
        T, N, F_in = x_seq.shape
        device = x_seq.device
        static_edge_index = static_edge_index.to(device)
        
        # 1. 前向传播计算 embeddings (为了得到正确的 g)
        # 这里必须完整跑一遍 forward 的流程才能拿到正确的 z_graph, z_hyper 从而算出 g
        # 我们可以复用 forward 代码，或者简化调用。为了稳健，我们手动跑一遍核心逻辑。
        
        x_emb = self.input_proj(x_seq)
        
        graph_feats_list = []
        hyper_feats_list = []
        
        # 用于累积结构
        accum_weights = None
        accum_hyper_adj = torch.zeros((N, N), device=device, dtype=torch.float32)
        
        for t in range(T):
            x_t = x_emb[t]
            
            # --- Graph Branch ---
            w_pred = self.edge_predictor(x_t, static_edge_index)
            if accum_weights is None:
                accum_weights = torch.zeros_like(w_pred)
            accum_weights += w_pred
            
            # (为了 g 的计算，需要跑卷积)
            h_graph = x_t
            for conv in self.graph_convs:
                h_graph = conv(h_graph, edge_index=static_edge_index, edge_weight=w_pred)
                h_graph = F.relu(h_graph)
            graph_feats_list.append(h_graph)
            
            # --- Hypergraph Branch ---
            # 我们需要计算有效邻接矩阵 A_hyp = mean_scales( Dv^-0.5 H De^-1 H^T Dv^-0.5 )
            current_step_hyper_adj = torch.zeros((N, N), device=device, dtype=torch.float32)
            
            if len(self.config.num_scales) > 0:
                scale_outputs = []
                for idx, k in enumerate(self.config.num_scales):
                    builder = self.hyper_builders[idx]
                    H_sparse = builder(x_t) # [N, N] sparse, indices=[row(node), col(edge)]
                    
                    # 1. 计算这一尺度的 Hyper Adjacency
                    # H_sparse 实际上是 Incidence Matrix H (N x E)
                    # 这里的 builder 实现稍微特殊，它返回的是 N x N 的 sparse tensor 
                    # 其中 col_idx 实际上代表了 "以该列节点为中心的超边"。
                    # 所以 H 矩阵的维度是 N x N_hyperedges (这里 N_hyperedges = N)
                    
                    # 按照 HGNN 定义: A = H H^T (假设 W_e=I)
                    # 标准化: D_v^-0.5 H H^T D_v^-0.5 (假设 D_e = k+1 是常数，并在特征中处理了)
                    # 我们这里显式构建:
                    
                    # H_sparse: [N, N_edges]
                    # H_dense = H_sparse.to_dense() # 安全起见转 dense，防止 sparse mm 问题
                    # A_scale = H_dense @ H_dense.t()
                    
                    # 优化: 我们可以直接用 sparse mm
                    # A_scale = torch.sparse.mm(H_sparse, H_sparse.t()) 
                    # 但 torch.sparse.mm 需要 (Sparse, Dense)。
                    # 所以 H_dense 是必须的。对于 N=5000，H_dense 只有 25M 个元素，显存占用极小。
                    
                    H_dense = H_sparse.to_dense() 
                    
                    # 计算度 D_v (行和)
                    Dv = H_dense.sum(dim=1).clamp(min=1e-6)
                    Dv_inv_sqrt = torch.diag(torch.pow(Dv, -0.5))
                    
                    # 计算度 D_e (列和，即超边大小，通常是 k+1)
                    De = H_dense.sum(dim=0).clamp(min=1e-6)
                    De_inv = torch.diag(torch.pow(De, -1.0))
                    
                    # A_norm = Dv^-0.5 @ H @ De^-1 @ H^T @ Dv^-0.5
                    # 这一项就是 Module 3 中 L^(2) 对应的 "Adjacency Part"
                    term1 = Dv_inv_sqrt @ H_dense # [N, E]
                    term2 = De_inv @ term1.t()    # [E, N]
                    A_scale = term1 @ term2       # [N, N]
                    
                    current_step_hyper_adj += A_scale
                    
                    # (为了 g 的计算，需要跑卷积)
                    h_hyper_scale = x_t
                    scale_convs = self.hyper_convs_per_scale[idx]
                    for conv in scale_convs:
                        h_hyper_scale = conv(h_hyper_scale, H_sparse)
                        h_hyper_scale = F.relu(h_hyper_scale)
                    scale_outputs.append(h_hyper_scale)
                
                # 平均不同尺度的邻接矩阵
                current_step_hyper_adj /= len(self.config.num_scales)
                
                h_hyper_t = torch.stack(scale_outputs, dim=0).mean(dim=0)
            else:
                h_hyper_t = torch.zeros_like(x_t)
            
            accum_hyper_adj += current_step_hyper_adj
            hyper_feats_list.append(h_hyper_t)

        # Time Averaging
        avg_weights = accum_weights / T
        avg_hyper_adj = accum_hyper_adj / T
        
        # Calculate g
        graph_seq = torch.stack(graph_feats_list, dim=0).permute(1, 0, 2)
        hyper_seq = torch.stack(hyper_feats_list, dim=0).permute(1, 0, 2)
        z_g = self.graph_temporal_encoder(graph_seq)
        z_h = self.hyper_temporal_encoder(hyper_seq)
        
        g, _ = self.gating(z_g, z_h)
        
        return avg_weights, avg_hyper_adj, g.squeeze(-1)