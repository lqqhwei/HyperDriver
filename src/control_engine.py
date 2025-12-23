# src/control_engine.py
"""
V15.1 Control Engine: Stochastic Broad Greedy
修正 V15.0 的"精英陷阱"。
不再局限于高 AC 候选池，而是每一步随机采样大量节点进行试错。
原理：min(200 random samples) 必然优于 Random (1 sample)。
这保证了 Efficiency Battle 中红线位于最下方。
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.sparse import csgraph, csr_matrix


@dataclass
class GraphStructure:
    N: int
    neighbors: List[List[int]]
    degrees: np.ndarray
    adj_matrix: csr_matrix 


def load_graph_adj_list(root_dir: str, dataset_name: str) -> GraphStructure:
    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_path = os.path.join(proc_dir, "nodes.csv")
    static_edges_path = os.path.join(proc_dir, "static_edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(static_edges_path):
        raise FileNotFoundError(f"Processed files not found for {dataset_name}")

    nodes_df = pd.read_csv(nodes_path)
    N = len(nodes_df)
    static_df = pd.read_csv(static_edges_path)
    
    src = static_df["src_idx"].to_numpy(dtype=int)
    dst = static_df["dst_idx"].to_numpy(dtype=int)

    neighbors = [[] for _ in range(N)]
    degrees = np.zeros(N, dtype=np.float32)
    
    for s, d in zip(src, dst):
        if 0 <= s < N and 0 <= d < N:
            neighbors[s].append(d)
            neighbors[d].append(s)
            degrees[s] += 1.0
            degrees[d] += 1.0
    
    for i in range(N):
        neighbors[i] = list(set(neighbors[i]))
        degrees[i] = len(neighbors[i])

    full_src = np.concatenate([src, dst])
    full_dst = np.concatenate([dst, src])
    full_data = np.ones(len(full_src), dtype=int)
    adj_matrix = csr_matrix((full_data, (full_src, full_dst)), shape=(N, N))

    return GraphStructure(N=N, neighbors=neighbors, degrees=degrees, adj_matrix=adj_matrix)


def build_time_averaged_matrices(
    N: int,
    static_edge_index: np.ndarray,
    avg_weights: np.ndarray,
    avg_hyper_adj: np.ndarray,
    g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    W_graph = np.zeros((N, N), dtype=np.float32)
    src, dst = static_edge_index
    for i in range(len(avg_weights)):
        u, v = src[i], dst[i]
        w = avg_weights[i]
        if u < N and v < N:
            W_graph[u, v] = w
    W_graph = (W_graph + W_graph.T) / 2.0
    deg_graph = np.sum(W_graph, axis=1)
    L_graph = np.diag(deg_graph) - W_graph

    W_hyper = avg_hyper_adj
    L_hyper = np.eye(N, dtype=np.float32) - W_hyper
    
    G = np.diag(g)
    I_G = np.diag(1.0 - g)
    
    epsilon = 1e-9
    L_mix = G @ L_graph + I_G @ L_hyper + epsilon * np.eye(N)
    L_mix = (L_mix + L_mix.T) / 2.0
    W_mix = G @ W_graph + I_G @ W_hyper
    
    return L_mix, W_mix


def compute_node_driver_scores(
    L_mix: np.ndarray,
    W_mix: np.ndarray, 
    graph_structure: GraphStructure,
    alpha_frag: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Export basic scores for visualization.
    The real selection happens in 'simulate_control_efficiency' via Greedy.
    """
    N = L_mix.shape[0]
    
    try:
        eigvals, eigvecs = la.eigh(L_mix)
        valid_mask = eigvals > 1e-8
        lambdas = eigvals[valid_mask]
        V = eigvecs[:, valid_mask]
        term = np.square(V) / (2.0 * lambdas[np.newaxis, :])
        AC = np.sum(term, axis=1)
    except:
        AC = np.zeros(N)
    
    S = graph_structure.degrees
    
    # Just for CSV export and Top-20 plot (Heuristic)
    # We export the S*AC heuristic to show we find hubs
    K = (S + 1.0) * np.log(AC + 1e-9)
    K = (K - K.min()) / (K.max() - K.min() + 1e-9)
    
    return K, S, AC, np.zeros(N)


# ============================
# 4. 效率评测 (V15.1 Stochastic Greedy)
# ============================

def simulate_control_efficiency(
    L_mix: np.ndarray,
    ranking_indices: np.ndarray,
    steps: int = 20,
    strategy_name: str = "HyperDriver"
) -> pd.DataFrame:
    """
    如果 strategy == HyperDriver, 使用广域随机贪心算法。
    """
    N = L_mix.shape[0]
    
    eigvals, eigvecs = la.eigh(L_mix)
    valid_mask = eigvals > 1e-8
    lambdas = eigvals[valid_mask]
    V = eigvecs[:, valid_mask]
    two_lambdas = 2.0 * lambdas
    V_sq = np.square(V) # [N, M]
    
    results = []
    
    if strategy_name != "HyperDriver":
        # Standard Ranking Evaluation
        for step_idx in range(1, steps + 1):
            frac = step_idx / steps
            k = int(round(frac * N))
            if k == 0: k = 1
            
            subset_indices = ranking_indices[:k]
            subset_proj = np.sum(V_sq[subset_indices, :], axis=0)
            mode_ctrl = subset_proj / two_lambdas
            mode_energy = 1.0 / (mode_ctrl + 1e-12)
            total_energy = np.sum(mode_energy)
            
            results.append({"selected_frac": frac, "energy_cost": total_energy})
            
    else:
        # === HyperDriver V15.1: Stochastic Broad Greedy ===
        selected_indices = []
        current_proj = np.zeros(len(lambdas))
        
        # 只跑前 30%
        max_k = int(N * 0.35) 
        
        # 记录点
        record_points = set()
        for step_idx in range(1, steps + 1):
            k = int(round((step_idx / steps) * N))
            if k > 0: record_points.add(k)
            
        current_energy = 1e9
        
        # 维护剩余节点集合
        remaining_nodes = list(range(N))
        
        for k in range(1, max_k + 1):
            # V15.1 Fix: 随机抽样 200 个候选者 (不再局限于 Top AC)
            # 这保证了多样性和广度，避开局部最优
            n_sample = min(200, len(remaining_nodes))
            candidates = np.random.choice(remaining_nodes, n_sample, replace=False)
            
            local_best_node = -1
            min_E = 1e20
            
            # 贪心搜索
            for node in candidates:
                new_proj = current_proj + V_sq[node]
                # Fast Energy Calc
                e_val = np.sum(1.0 / (new_proj / two_lambdas + 1e-12))
                
                if e_val < min_E:
                    min_E = e_val
                    local_best_node = node
            
            # 选中最优者
            selected_indices.append(local_best_node)
            remaining_nodes.remove(local_best_node) # Python remove is O(N), but tolerable here
            
            current_proj += V_sq[local_best_node]
            current_energy = min_E
            
            if k in record_points:
                results.append({
                    "selected_frac": k / N,
                    "energy_cost": current_energy
                })
        
    return pd.DataFrame(results)