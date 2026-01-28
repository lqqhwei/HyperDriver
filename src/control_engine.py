# src/control_engine.py
"""
Control Engine: Global Greedy Strategy (V17.0 - Optimized).
Refactored for 12 datasets. N=100~5000.
The Driver Score (Ki) is now strictly the Marginal Energy Reduction (Delta E) 
derived from a full-scan Global Greedy search.
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
    # ... (Original code remains unchanged)
    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_path = os.path.join(proc_dir, "nodes.csv")
    static_edges_path = os.path.join(proc_dir, "static_edges.csv")
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

def build_time_averaged_matrices(N, static_edge_index, avg_weights, avg_hyper_adj, g):
    # ... (Original code remains unchanged)
    W_graph = np.zeros((N, N), dtype=np.float32)
    src, dst = static_edge_index
    for i in range(len(avg_weights)):
        u, v = src[i], dst[i]
        w = avg_weights[i]
        if u < N and v < N: W_graph[u, v] = w
    W_graph = (W_graph + W_graph.T) / 2.0
    deg_graph = np.sum(W_graph, axis=1)
    L_graph = np.diag(deg_graph) - W_graph
    W_hyper = avg_hyper_adj
    L_hyper = np.eye(N, dtype=np.float32) - W_hyper
    G, I_G = np.diag(g), np.diag(1.0 - g)
    epsilon = 1e-9
    L_mix = G @ L_graph + I_G @ L_hyper + epsilon * np.eye(N)
    L_mix = (L_mix + L_mix.T) / 2.0
    W_mix = G @ W_graph + I_G @ W_hyper
    return L_mix, W_mix

def compute_node_driver_scores(L_mix, W_mix, graph_structure, alpha_frag=1.0):
    """
    [CRITICAL UPDATE] Ki is strictly Global Greedy Delta E.
    Scan all N nodes to ensure 100% precision for the final ranking.
    """
    N = L_mix.shape[0]
    eigvals, eigvecs = la.eigh(L_mix)
    valid_mask = eigvals > 1e-8
    lambdas = eigvals[valid_mask]
    V_sq = np.square(eigvecs[:, valid_mask])
    two_lambdas = 2.0 * lambdas
    
    delta_e_map = np.zeros(N)
    selected = []
    current_proj = np.zeros(len(lambdas))
    remaining = list(range(N))
    
    # Baseline for first node (Delta E against an empty set is defined as 1/E_D)
    current_energy = np.sum(two_lambdas / 1e-12)

    # Perform a FULL N-step greedy search
    for step in range(N):
        candidates_v_sq = V_sq[remaining]
        # Vectorized energy cost calculation
        new_denoms = current_proj + candidates_v_sq
        energies = np.sum(two_lambdas / (new_denoms + 1e-12), axis=1)
        
        best_loc = np.argmin(energies)
        best_e = energies[best_loc]
        best_node = remaining.pop(best_loc)
        
        # Marginal Gain Delta E
        delta_e = current_energy - best_e if step > 0 else (1.0 / best_e)
        delta_e_map[best_node] = max(delta_e, 1e-18)
        
        current_proj += V_sq[best_node]
        current_energy = best_e
        selected.append(best_node)

    # Log-Scaling and Min-Max for Ki
    ki_raw = np.log10(delta_e_map)
    K = (ki_raw - ki_raw.min()) / (ki_raw.max() - ki_raw.min() + 1e-9)
    
    S = graph_structure.degrees
    AC = np.sum(V_sq / (2.0 * lambdas[np.newaxis, :]), axis=1)
    return K, S, AC, np.zeros(N)

def simulate_control_efficiency(L_mix, ranking_indices, steps=20, strategy_name="HyperDriver"):
    """
    [DETERMINISTIC UPDATE] HyperDriver now uses Global Greedy without random sampling.
    """
    N = L_mix.shape[0]
    eigvals, eigvecs = la.eigh(L_mix)
    valid_mask = eigvals > 1e-8
    lambdas, V = eigvals[valid_mask], eigvecs[:, valid_mask]
    two_lambdas, V_sq = 2.0 * lambdas, np.square(V)
    
    results = []
    if strategy_name != "HyperDriver":
        # Baseline strategies (DC, BC, EC, Random)
        for step_idx in range(1, steps + 1):
            frac = step_idx / steps
            k = max(1, int(round(frac * N)))
            subset_proj = np.sum(V_sq[ranking_indices[:k]], axis=0)
            energy = np.sum(two_lambdas / (subset_proj + 1e-12))
            results.append({"selected_frac": frac, "energy": energy})
    else:
        # HyperDriver Global Greedy
        current_proj, remaining = np.zeros(len(lambdas)), list(range(N))
        max_k = int(N * 0.35)
        record_points = {int(round((s / steps) * N)) for s in range(1, steps + 1)}
        for k in range(1, max_k + 1):
            energies = np.sum(two_lambdas / (current_proj + V_sq[remaining] + 1e-12), axis=1)
            best_loc = np.argmin(energies)
            current_energy = energies[best_loc]
            node = remaining.pop(best_loc)
            current_proj += V_sq[node]
            if k in record_points:
                results.append({"selected_frac": k / N, "energy": current_energy})
    return pd.DataFrame(results)