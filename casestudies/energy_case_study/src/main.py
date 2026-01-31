# resource/energy_case_study/src/main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Import modules
from data_loader import DataLoader
from utils_graph import SubgraphSelector
from methods import DriverSelector
from control_energy import ControlEnergyCalculator

# Configuration (Aligning with your structure)
ROOT = Path(__file__).resolve().parents[3] # Assuming d:/HD/resource/energy_case_study/src/main.py
OUTPUT_DIR = ROOT / "resource/energy_case_study/output"
STATIC_DIR = ROOT / "data/Yu"
# [Deleted] LABEL_DIR is no longer needed

TOP_K = 10               # Select 10 driving nodes
RANDOM_RUNS = 20         # Random baseline runs

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def main():
    print("========== Case Study: Energy Efficiency ==========")
    setup_dirs()
    
    # 1. Load Data
    # [Modified] Removed LABEL_DIR argument. Now only loads the graph structure.
    loader = DataLoader(str(STATIC_DIR))
    
    # [FIXED] load_static_graph returns ONLY the NetworkX Graph object 'G'.
    G_full = loader.load_static_graph()
    
    print(f"Graph loaded. Nodes: {len(G_full.nodes())}, Edges: {len(G_full.edges())}")
    
    # 2. Extract Subgraph
    # SubgraphSelector generates the Adjacency Matrix (A_sub) internally for the selected subgraph
    sub_selector = SubgraphSelector(G_full, target_size=(50, 80))
    G_sub, A_sub, sub_nodes, sub_idx_map = sub_selector.get_best_subgraph()
    
    print(f"Subgraph Selected: N={len(sub_nodes)} nodes")
    
    # 3. Initialize Selectors and Calculator
    driver_selector = DriverSelector(G_sub, A_sub, sub_nodes)
    energy_calc = ControlEnergyCalculator(A_sub)
    
    results = {}
    selection_map = {
        'Protein': sub_nodes,
        'HyperDriver': np.zeros(len(sub_nodes), dtype=int),
        'Degree': np.zeros(len(sub_nodes), dtype=int),
        'Random': np.zeros(len(sub_nodes), dtype=int)
    }

    # --- Strategy A: HyperDriver Global Greedy ---
    print(f"[Run] HyperDriver Global Greedy (Top {TOP_K})...")
    idx_hd, _ = driver_selector.select_hyperdriver(TOP_K)
    e_hd = energy_calc.compute_energy(idx_hd)
    results['HyperDriver'] = e_hd
    for idx in idx_hd: selection_map['HyperDriver'][idx] = 1

    # --- Strategy B: Degree Centrality ---
    print(f"[Run] Degree Centrality (Top {TOP_K})...")
    idx_deg, _ = driver_selector.select_degree(TOP_K)
    e_deg = energy_calc.compute_energy(idx_deg)
    results['Degree'] = e_deg
    for idx in idx_deg: selection_map['Degree'][idx] = 1

    # --- Strategy C: Random (Averaged) ---
    print(f"[Run] Random Selection ({RANDOM_RUNS} runs)...")
    random_energies = []
    # Just to record one sample selection for CSV
    sample_idx = [] 
    
    for i in tqdm(range(RANDOM_RUNS)):
        idx_rnd, _ = driver_selector.select_random(TOP_K, seed=i)
        e = energy_calc.compute_energy(idx_rnd)
        if e != float('inf'):
            random_energies.append(e)
        if i == 0: sample_idx = idx_rnd

    avg_random = np.mean(random_energies) if random_energies else float('inf')
    results['Random'] = avg_random
    for idx in sample_idx: selection_map['Random'][idx] = 1

    # 4. Save & Plot
    print("\nResults Summary (Energy):")
    for m, e in results.items():
        print(f"  {m}: {e:.4e}")

    # Save Selection CSV
    df_sel = pd.DataFrame(selection_map)
    df_sel.to_csv(os.path.join(OUTPUT_DIR, "selection_nodes.csv"), index=False)
    
    # Save Results CSV
    df_res = pd.DataFrame(list(results.items()), columns=['Method', 'Energy'])
    df_res.to_csv(os.path.join(OUTPUT_DIR, "energy_results.csv"), index=False)
    
    print("\n[SUCCESS] Exp Case Study Completed.")

if __name__ == "__main__":
    main()