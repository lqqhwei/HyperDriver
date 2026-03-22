# eval_driver.py
"""
Step 4: HyperDriver Comprehensive Evaluation (Adaptive K-Means Version)
Features:
1. Integration with Global Greedy Engine.
2. [FINAL ADAPTIVE SOLUTION] Key Driver Identification:
   - Uses 1D K-Means Clustering (k=2) to separate 'Drivers' from 'Non-Drivers'.
   - Adaptive: Robust to curve shape (convex/concave), relies on score distribution.
   - Solves the 'select all' issue of Kneedle on concave curves.
"""

import argparse
import os
import random
import traceback
from typing import Dict

import numpy as np
import pandas as pd
import scipy.linalg as la
import torch

from src.data_utils import load_datasets_config
from src.hyper_driver import HyperDriver, HyperDriverConfig
from src.control_engine import (
    load_graph_adj_list,
    build_time_averaged_matrices,
    compute_node_driver_scores,
    simulate_control_efficiency,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_baseline_scores(root_dir: str, dataset_name: str, N: int) -> Dict[str, np.ndarray]:
    base_dir = os.path.join(root_dir, "results", dataset_name, "baselines")
    methods = {"dc": "dc_scores.csv", "bc": "bc_scores.csv", "ec": "ec_scores.csv"} 
    loaded = {}
    for m, fname in methods.items():
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if len(df) == N: loaded[m] = df[m].to_numpy(dtype=float)
            except: pass
    return loaded

# Helper: 1D K-Means Clustering (k=2)
def find_kmeans_threshold(scores: np.ndarray) -> float:
    """
    Splits scores into High/Low clusters using iterative K-Means.
    Returns the threshold value (midpoint between final centroids).
    """
    # Initialize centroids with min and max
    c1 = np.min(scores) # Low center
    c2 = np.max(scores) # High center
    
    # Iterate to converge (usually takes <10 steps)
    for _ in range(20):
        # 1. Assign points to nearest centroid
        d1 = np.abs(scores - c1)
        d2 = np.abs(scores - c2)
        
        group_low = scores[d1 <= d2]
        group_high = scores[d1 > d2]
        
        # Safety check: if one group is empty, break
        if len(group_low) == 0 or len(group_high) == 0:
            break
            
        # 2. Update centroids
        new_c1 = np.mean(group_low)
        new_c2 = np.mean(group_high)
        
        # Check convergence
        if np.abs(new_c1 - c1) < 1e-6 and np.abs(new_c2 - c2) < 1e-6:
            c1, c2 = new_c1, new_c2
            break
        c1, c2 = new_c1, new_c2
    
    # The decision boundary is the midpoint
    threshold = (c1 + c2) / 2.0
    return threshold

# Function: Adaptive K-Means Identification
def identify_key_drivers_kmeans(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies key drivers using K-Means Clustering with a Sparsity Guardrail.
    Strategy:
    1. Try K-Means (Adaptive).
    2. If K-Means selects > 60% of nodes (Degenerate Case), fallback to Mean + Std.
    """
    if "driver_score" not in nodes_df.columns:
        return pd.DataFrame()

    # 1. Sort by driver_score (Descending)
    valid_df = nodes_df[nodes_df["driver_score"] > -np.inf].copy()
    sorted_df = valid_df.sort_values("driver_score", ascending=False).reset_index(drop=True)
    N = len(sorted_df)
    
    if sorted_df.empty:
        return pd.DataFrame()
    
    scores = sorted_df["driver_score"].to_numpy()
    
    # --- Strategy A: K-Means ---
    threshold_km = find_kmeans_threshold(scores)
    
    # Check selection ratio
    n_selected_km = np.sum(scores >= threshold_km)
    ratio = n_selected_km / N
    
    # --- Strategy B: Fallback (Mean + Std) ---
    # Trigger if K-Means selects too many (>60%) or too few (0)
    if ratio > 0.60 or n_selected_km == 0:
        print(f"  [WARN] K-Means degenerate (ratio={ratio:.2f}). Switching to Mean+Std.")
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        # Fallback threshold: Average + 1.0 Standard Deviation (Approx Top 16% in normal dist)
        threshold = mean_s + std_s
    else:
        threshold = threshold_km

    # 4. Slice
    key_drivers = sorted_df[sorted_df["driver_score"] >= threshold].copy()
    
    # 5. Format Output
    key_drivers["rank"] = key_drivers.index + 1
    output_cols = ["rank", "protein", "driver_score"]
    
    return key_drivers[output_cols]

def eval_dataset(root_dir: str, dataset_name: str, args):
    print(f"\n========== Evaluating: {dataset_name} ==========")
    device = get_device()
    
    ckpt_path = os.path.join(root_dir, "checkpoints", f"{dataset_name}_full.pt")
    res_dir = os.path.join(root_dir, "results", dataset_name, "full") 
    os.makedirs(res_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Full model not found: {ckpt_path}")
        return

    proc_dir = os.path.join(root_dir, "processed", dataset_name)
    nodes_df = pd.read_csv(os.path.join(proc_dir, "nodes.csv"))
    static_df = pd.read_csv(os.path.join(proc_dir, "static_edges.csv"))
    N = len(nodes_df)
    
    time_cols = sorted([c for c in nodes_df.columns if c.startswith("t") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    if time_cols:
        x_seq = torch.stack([torch.tensor(nodes_df[c].values, dtype=torch.float32) for c in time_cols]).unsqueeze(-1).to(device)
    else:
        x_seq = torch.zeros(1, N, 1, device=device)
    edge_index = torch.tensor(np.stack([static_df["src_idx"], static_df["dst_idx"]]), dtype=torch.long, device=device)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
    config = HyperDriverConfig(**checkpoint["config"])
    model = HyperDriver(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # [1/6] Extracting Learned Structures
    print("[1/6] Extracting Learned Structures...")
    with torch.no_grad():
        avg_weights, avg_hyper_adj, g = model.get_consensus_structure(x_seq, edge_index)
        
    avg_weights_np, avg_hyper_adj_np = avg_weights.cpu().numpy(), avg_hyper_adj.cpu().numpy()
    g_np, edge_index_np = g.cpu().numpy(), edge_index.cpu().numpy()

    battle_strategies = []

    # [2/6] Baseline Simulations
    print("[2/6] Simulating Random & Baselines...")
    rng = np.random.default_rng(args.seed)
    L_mix_full, W_mix_full = build_time_averaged_matrices(N, edge_index_np, avg_weights_np, avg_hyper_adj_np, g_np)
    
    rand_order = np.argsort(rng.random(N))
    battle_strategies.append(simulate_control_efficiency(L_mix_full, rand_order, args.attack_steps, "Random").assign(strategy="Random"))

    baselines = load_baseline_scores(root_dir, dataset_name, N)
    for b_name, scores in baselines.items():
        order = np.argsort(-scores)
        battle_strategies.append(simulate_control_efficiency(L_mix_full, order, args.attack_steps, b_name.upper()).assign(strategy=b_name.upper()))

    # [3/6] HyperDriver with Global Greedy
    print("[3/6] Simulating HyperDriver via Global Greedy...")
    graph_struct = load_graph_adj_list(root_dir, dataset_name)
    K_score, S, AC, F = compute_node_driver_scores(L_mix_full, W_mix_full, graph_struct)
    
    nodes_df["driver_score"] = K_score
    nodes_df["score_S"] = S
    nodes_df["score_AC"] = AC
    nodes_df.to_csv(os.path.join(res_dir, "node_scores.csv"), index=False)

    # --- [FINAL] Key Driver Identification (Adaptive K-Means) ---
    print(f"[4/6] Identifying Key Drivers using Adaptive K-Means Clustering...")
    keys_dir = os.path.join(root_dir, "results", dataset_name, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    # Adaptive K-Means
    key_drivers = identify_key_drivers_kmeans(nodes_df)
    
    key_drivers.to_csv(os.path.join(keys_dir, "key_driver_nodes.csv"), index=False)
    
    print(f"[INFO] K-Means identified {len(key_drivers)} key drivers ({len(key_drivers)/N*100:.1f}% of total).")
    # ------------------------------------------------

    df_hd = simulate_control_efficiency(L_mix_full, None, args.attack_steps, "HyperDriver")
    df_hd["strategy"] = "HyperDriver"
    battle_strategies.append(df_hd)

    # [5/6] Refined Ablations
    print("[5/6] Simulating Ablations (Inference-Time)...")
    rank_order_ac = np.argsort(-AC)
    battle_strategies.append(simulate_control_efficiency(L_mix_full, rank_order_ac, args.attack_steps, "Ranking").assign(strategy="w/o Greedy"))

    g_ones = np.ones_like(g_np)
    L_mix_graph, _ = build_time_averaged_matrices(N, edge_index_np, avg_weights_np, avg_hyper_adj_np, g_ones)
    battle_strategies.append(simulate_control_efficiency(L_mix_graph, None, args.attack_steps, "HyperDriver").assign(strategy="w/o Hypergraph"))

    static_weights = np.ones_like(avg_weights_np)
    L_mix_static, _ = build_time_averaged_matrices(N, edge_index_np, static_weights, avg_hyper_adj_np, g_np)
    battle_strategies.append(simulate_control_efficiency(L_mix_static, None, args.attack_steps, "HyperDriver").assign(strategy="w/o Dynamics"))

    # [6/6] Saving Battle Results
    print("[6/6] Saving Efficiency Data...")
    final_df = pd.concat(battle_strategies, ignore_index=True)
    final_df.to_csv(os.path.join(res_dir, "energy_battle.csv"), index=False)

    print(f"[INFO] Evaluation complete for {dataset_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--attack_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # No more threshold needed!
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets_config(os.path.join(root, "conf", "datasets.json")) if args.dataset == "all" else [args.dataset]
    set_seed(args.seed)
    for ds in datasets:
        try:
            eval_dataset(root, ds, args)
        except Exception as e:
            print(f"[ERROR] Failed on {ds}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()