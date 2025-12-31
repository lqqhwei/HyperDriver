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
    HyperDriver Model Configuration (V4.2 Optimized: Sparse + Residual)
    """
    in_feats: int                  
    hidden_dim: int = 64           
    num_graph_layers: int = 2      
    num_hyper_layers: int = 1      
    num_time_layers: int = 1       
    num_classes: int = 1           
    # [Optimization] Only retain fine-scale values ​​to reduce computational load and GPU memory usage.
    num_scales: List[int] = field(default_factory=lambda: [5, 10, 15]) 
    # [Optimization] Improve Dropout
    dropout: float = 0.5


class HyperDriver(nn.Module):
    """
    HyperDriver V4.2 Main model
    """

    def __init__(self, config: HyperDriverConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # -------- Feature projection --------
        self.input_proj = nn.Linear(config.in_feats, H)

        # -------- Module 1: Learning with Animated Graphs (Student) --------
        self.edge_predictor = EdgePredictor(in_channels=H, hidden_dim=H // 2)

        self.graph_convs = nn.ModuleList()
        for i in range(config.num_graph_layers):
            self.graph_convs.append(GraphConvLayer(H, H))

        self.graph_temporal_encoder = TemporalEncoder(H, H, config.num_time_layers)

        # -------- Module 2: Multi-scale Dynamic Hypergraph (Sparse) --------
        self.hyper_builders = nn.ModuleList()
        self.hyper_convs_per_scale = nn.ModuleList()

        for k in config.num_scales:
            self.hyper_builders.append(DynamicHypergraphBuilder(k_neighbors=k))
            scale_layers = nn.ModuleList()
            for j in range(config.num_hyper_layers):
                scale_layers.append(HypergraphConvLayer(H, H))
            self.hyper_convs_per_scale.append(scale_layers)

        self.hyper_temporal_encoder = TemporalEncoder(H, H, config.num_time_layers)

        # -------- Module 3: Gating and Classification (Residual) --------
        self.gating = NodeGatingLayer(in_channels=H)
        self.hyper_residual_proj = nn.Linear(H, H) # Used for residual alignment

        # Here, alpha is a learnable parameter that controls the initial proportion of the residual.
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
        Standard training forward propagation
        """
        T, N, F_in = x_seq.shape
        device = x_seq.device
        static_edge_index = static_edge_index.to(device)

        # 1. Feature projection
        x_emb = self.input_proj(x_seq)

        graph_feats_list = []
        hyper_feats_list = []
        predicted_weights_list = []

        # 2. Loop by time step
        for t in range(T):
            x_t = x_emb[t]

            # ========== Module 1: Dynamic Graph Branches ==========
            w_pred = self.edge_predictor(x_t, static_edge_index)
            predicted_weights_list.append(w_pred)

            h_graph = x_t
            for conv in self.graph_convs:
                h_graph = conv(h_graph, edge_index=static_edge_index, edge_weight=w_pred)
                h_graph = F.relu(h_graph)
                h_graph = self.dropout(h_graph)
            graph_feats_list.append(h_graph)

            # ========== Module 2: Dynamic Hypergraph Branches (Sparse) ==========
            if len(self.config.num_scales) > 0:
                scale_outputs = []
                for idx, k in enumerate(self.config.num_scales):
                    # Builder The returned value is a sparse tensor. H_sparse
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

        # 3. Time-series aggregation
        # [T, N, H] -> [N, T, H]
        graph_seq_tensor = torch.stack(graph_feats_list, dim=0).permute(1, 0, 2)
        z_graph = self.graph_temporal_encoder(graph_seq_tensor)

        hyper_seq_tensor = torch.stack(hyper_feats_list, dim=0).permute(1, 0, 2)
        z_hyper = self.hyper_temporal_encoder(hyper_seq_tensor)

        # 4. Module 3: Residual Fusion
        # g is used here to generate the gating mask (N, 1).
        g, _ = self.gating(z_graph, z_hyper) 
        
        z_hyper_proj = self.hyper_residual_proj(z_hyper)
        # Explicitly using sigmoid(alpha) to control the blending ratio ensures stability during the initial training phase.
        z_mix = z_graph + torch.sigmoid(self.alpha) * 0.1 * z_hyper_proj

        # 5. Classification Head
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
        [NEW] Retrieves the "time-averaged hybrid structure" for Average Energy Controllability analysis.
        Returns:
        avg_weights: (E_static, ) Mean of dynamic weights on the static skeleton (mean of w_pred)
        avg_hyper_adj: (N, N) Mean of effective adjacency matrix of hypergraph branches (mean of A_hyper)
        g: (N, ) Node-level gating value (time-invariant structure preference)
        
        Logic:
        1. Iterate through the time steps, accumulating w_pred.
        2. Iterate through the time steps, construct H_sparse, and calculate the normalized adjacency matrix A = Dv^-0.5 H De^-1 H^T Dv^-0.5 corresponding to L_hyper.
        Although H is sparse, for convenience in subsequent spectral decomposition, we accumulate and return an N x N matrix in dense format here.
        (For datasets with N <= 5000, 5000^2 * 4B ≈ 100MB is perfectly acceptable)
        3. Return g.
        """
        T, N, F_in = x_seq.shape
        device = x_seq.device
        static_edge_index = static_edge_index.to(device)
        
        # 1. Forward propagation computes embeddings (in order to obtain the correct g).
        # To obtain the correct z_graph and z_hyper, we must run the entire forward process here, and then calculate g.
        # We can reuse the forward code or simplify the calls. For robustness, we can manually run the core logic once.
        
        x_emb = self.input_proj(x_seq)
        
        graph_feats_list = []
        hyper_feats_list = []
        
        # Used for cumulative structures
        accum_weights = None
        accum_hyper_adj = torch.zeros((N, N), device=device, dtype=torch.float32)
        
        for t in range(T):
            x_t = x_emb[t]
            
            # --- Graph Branch ---
            w_pred = self.edge_predictor(x_t, static_edge_index)
            if accum_weights is None:
                accum_weights = torch.zeros_like(w_pred)
            accum_weights += w_pred
            
            # (Convolution is required to calculate g)
            h_graph = x_t
            for conv in self.graph_convs:
                h_graph = conv(h_graph, edge_index=static_edge_index, edge_weight=w_pred)
                h_graph = F.relu(h_graph)
            graph_feats_list.append(h_graph)
            
            # --- Hypergraph Branch ---
            # We need to calculate the effective adjacency matrix A_hyp = mean_scales( Dv^-0.5 H De^-1 H^T Dv^-0.5 )
            current_step_hyper_adj = torch.zeros((N, N), device=device, dtype=torch.float32)
            
            if len(self.config.num_scales) > 0:
                scale_outputs = []
                for idx, k in enumerate(self.config.num_scales):
                    builder = self.hyper_builders[idx]
                    H_sparse = builder(x_t) # [N, N] sparse, indices=[row(node), col(edge)]
                    
                    # 1. Calculate Hyper Adjacency at this scale
                    # H_sparse is actually the Incidence Matrix H (N x E)
                    # The builder implementation here is slightly special; it returns an N x N sparse tensor
                    # where col_idx actually represents the "hyperedge centered on the node in that column".
                    # Therefore, the dimension of the H matrix is ​​N x N_hyperedges (where N_hyperedges = N)
                    # According to the HGNN definition: A = H H^T (assuming W_e=I)
                    # Normalization: D_v^-0.5 H H^T D_v^-0.5 (assuming D_e = k+1 is a constant and processed in the features)
                    # We explicitly construct it here:
                    # H_sparse: [N, N_edges]
                    # H_dense = H_sparse.to_dense() # To convert to dense for safety, to prevent sparse mm problems
                    # A_scale = H_dense @ H_dense.t()
                    # Optimization: We can directly use sparse mm
                    # A_scale = torch.sparse.mm(H_sparse, H_sparse.t())
                    # But torch.sparse.mm requires (Sparse, Dense).
                    # Therefore, H_dense is necessary. For N=5000, H_dense has only 25M elements, resulting in minimal memory usage.
                    
                    H_dense = H_sparse.to_dense() 
                    
                    # Calculate the degree D_v (row sum).
                    Dv = H_dense.sum(dim=1).clamp(min=1e-6)
                    Dv_inv_sqrt = torch.diag(torch.pow(Dv, -0.5))
                    
                    # Calculate the degree D_e (column sum, i.e., the size of the hyperedge, usually k+1).
                    De = H_dense.sum(dim=0).clamp(min=1e-6)
                    De_inv = torch.diag(torch.pow(De, -1.0))
                    
                    # A_norm = Dv^-0.5 @ H @ De^-1 @ H^T @ Dv^-0.5
                    # This item corresponds to the "Adjacency Part" of L^(2) in Module 3.
                    term1 = Dv_inv_sqrt @ H_dense # [N, E]
                    term2 = De_inv @ term1.t()    # [E, N]
                    A_scale = term1 @ term2       # [N, N]
                    
                    current_step_hyper_adj += A_scale
                    
                    # (Convolution is required to calculate g)
                    h_hyper_scale = x_t
                    scale_convs = self.hyper_convs_per_scale[idx]
                    for conv in scale_convs:
                        h_hyper_scale = conv(h_hyper_scale, H_sparse)
                        h_hyper_scale = F.relu(h_hyper_scale)
                    scale_outputs.append(h_hyper_scale)
                
                # Average adjacency matrix at different scales
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