# eval_driver.py
"""
Step 4: HyperDriver Comprehensive Evaluation
Features:
1. Load only the "full" model (no longer searching for other variants).
2. Execute the Efficiency Battle (main experiment).
3. Execute the Ablation Study (ablation during inference).
"""

import argparse
import json
import os
import random
import traceback
from typing import Dict

import numpy as np
import pandas as pd
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

def eval_dataset(root_dir: str, dataset_name: str, args):
    print(f"\n========== Evaluating: {dataset_name} ==========")
    device = get_device()
    
    # [Clean] Load only full checkpoint
    ckpt_path = os.path.join(root_dir, "checkpoints", f"{dataset_name}_full.pt")
    res_dir = os.path.join(root_dir, "results", dataset_name, "full") 
    os.makedirs(res_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Full model not found: {ckpt_path}")
        return

    # Load Data
    from src.data_utils import get_dataset_paths
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

    # 1. Extracting Structure
    print("[1/6] Extracting Learned Structures...")
    with torch.no_grad():
        avg_weights, avg_hyper_adj, g = model.get_consensus_structure(x_seq, edge_index)
        
    avg_weights_np = avg_weights.cpu().numpy()
    avg_hyper_adj_np = avg_hyper_adj.cpu().numpy()
    g_np = g.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()

    battle_strategies = []

    # 2. Main Baselines
    print("[2/6] Simulating Random & Baselines...")
    rng = np.random.default_rng(args.seed)
    L_mix_full, _ = build_time_averaged_matrices(N, edge_index_np, avg_weights_np, avg_hyper_adj_np, g_np)
    
    # Random
    rand_order = np.argsort(rng.random(N))
    battle_strategies.append(simulate_control_efficiency(L_mix_full, rand_order, args.attack_steps, "Random").assign(strategy="Random"))

    # DC/BC/EC
    baselines = load_baseline_scores(root_dir, dataset_name, N)
    for b_name, scores in baselines.items():
        order = np.argsort(-scores)
        battle_strategies.append(simulate_control_efficiency(L_mix_full, order, args.attack_steps, b_name.upper()).assign(strategy=b_name.upper()))

    # 3. HyperDriver (Full)
    print("[3/6] Simulating HyperDriver (Full)...")
    # Save the scores for use in the Top-20 Plot.
    graph_struct = load_graph_adj_list(root_dir, dataset_name)
    _, W_mix_full = build_time_averaged_matrices(N, edge_index_np, avg_weights_np, avg_hyper_adj_np, g_np)
    K_score, S, AC, F = compute_node_driver_scores(L_mix_full, W_mix_full, graph_struct)
    
    nodes_df["driver_score"] = K_score
    nodes_df["score_S"] = S
    nodes_df["score_AC"] = AC
    nodes_df.to_csv(os.path.join(res_dir, "node_scores.csv"), index=False)

    # Running Greedy simulation
    df_hd = simulate_control_efficiency(L_mix_full, None, args.attack_steps, "HyperDriver")
    df_hd["strategy"] = "HyperDriver (Full)"
    battle_strategies.append(df_hd)

    # 4. Ablations
    print("[4/6] Simulating Ablations (Inference-Time)...")
    
    # w/o Greedy
    rank_order = np.argsort(-K_score)
    battle_strategies.append(simulate_control_efficiency(L_mix_full, rank_order, args.attack_steps, "Ranking").assign(strategy="w/o Greedy"))

    # w/o Hypergraph
    g_ones = np.ones_like(g_np)
    L_mix_graph, _ = build_time_averaged_matrices(N, edge_index_np, avg_weights_np, avg_hyper_adj_np, g_ones)
    battle_strategies.append(simulate_control_efficiency(L_mix_graph, None, args.attack_steps, "HyperDriver").assign(strategy="w/o Hypergraph"))

    # w/o Dynamics
    static_weights = np.ones_like(avg_weights_np)
    L_mix_static, _ = build_time_averaged_matrices(N, edge_index_np, static_weights, avg_hyper_adj_np, g_np)
    battle_strategies.append(simulate_control_efficiency(L_mix_static, None, args.attack_steps, "HyperDriver").assign(strategy="w/o Dynamics"))

    # 5. Save
    print("[5/6] Saving Results...")
    final_df = pd.concat(battle_strategies, ignore_index=True)
    final_df.to_csv(os.path.join(res_dir, "efficiency_battle.csv"), index=False)
    
    # 6. Hidden Drivers
    print("[6/6] Hidden Drivers Analysis...")
    if "essential" in nodes_df.columns:
        k_20 = int(N * 0.2)
        top_hd = np.argsort(-K_score)[:k_20]
        hidden = []
        ess = nodes_df["essential"].values
        for idx in top_hd:
            if ess[idx] == 0:
                hidden.append({"protein": nodes_df.iloc[idx]["protein"], "score": K_score[idx]})
        if hidden: pd.DataFrame(hidden).to_csv(os.path.join(res_dir, "hidden_drivers.csv"), index=False)

    print(f"[INFO] Evaluation finished for {dataset_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--attack_steps", type=int, default=20)
    parser.add_argument("--alpha_frag", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets_config(os.path.join(root, "conf", "datasets.json")) if args.dataset == "all" else [args.dataset]
    
    set_seed(args.seed)

    for ds in datasets:
        try:
            eval_dataset(root, ds, args)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()