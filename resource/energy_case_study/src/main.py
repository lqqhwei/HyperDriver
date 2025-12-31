import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Import previous modules
from data_loader import DataLoader
from utils_graph import SubgraphSelector
from methods import DriverSelector
from control_energy import ControlEnergyCalculator

# Configuration
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "resource/energy_case_study/output"
STATIC_DIR = ROOT / "data/Yu"
LABEL_DIR = ROOT / "data"
TOP_K = 10               # Select 10 driving nodes in the subgraph
RANDOM_RUNS = 20         # Run the randomized strategy 20 times and take the average.

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_results(results, subgraph_name="Case_Study"):
    """
    Plot an energy comparison histogram (Log Scale)
    """
    methods = list(results.keys())
    energies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if 'HyperDriver' in m else '#1f77b4' for m in methods]
    bars = plt.bar(methods, energies, color=colors, alpha=0.8, edgecolor='black')
    
    plt.ylabel('Required Control Energy ($J_{avg}$)', fontsize=12)
    plt.title(f'True Minimum Control Energy Comparison\n(Subgraph Size: {subgraph_name})', fontsize=14)
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.1, 
                 f'{yval:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path_png = os.path.join(OUTPUT_DIR, 'energy_comparison.png')
    plt.savefig(save_path_png, dpi=600)
    save_path_pdf = os.path.join(OUTPUT_DIR, 'energy_comparison.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    setup_dirs()
    print("Starting True Minimum Energy Case Study Experiment...")
    
    # 1. Load full image
    loader = DataLoader(STATIC_DIR, LABEL_DIR)
    G_full = loader.load_static_graph()
    
    # 2. Extracting the Case Study Subgraph
    selector = SubgraphSelector(G_full, target_size=(60, 120))
    G_sub, A_sub, sub_nodes, sub_map = selector.get_best_subgraph()
    
    print(f"\n Case Study Subgraph Selected: {len(sub_nodes)} nodes")
    
    # 3. Initialize the selector and calculator
    driver_selector = DriverSelector(G_sub, A_sub, sub_nodes)
    energy_calc = ControlEnergyCalculator(A_sub)
    
    results = {}
    
    # --- [Key change: Record the selected index] ---
    selection_map = {
        'Protein_Name': sub_nodes,
        'HyperDriver': np.zeros(len(sub_nodes), dtype=int),
        'Degree': np.zeros(len(sub_nodes), dtype=int),
        'Random': np.zeros(len(sub_nodes), dtype=int)
    }

    # ==========================
    # Method 1: HyperDriver
    # ==========================
    print(f"\n[1] Running HyperDriver...")
    idx_hd, names_hd = driver_selector.select_hyperdriver_proxy(TOP_K)
    energy_hd = energy_calc.compute_energy(idx_hd)
    results['HyperDriver'] = energy_hd
    selection_map['HyperDriver'][idx_hd] = 1 # Mark selected

    # ==========================
    # Method 2: Degree Centrality
    # ==========================
    print(f"\n[2] Running Degree Centrality...")
    idx_deg, names_deg = driver_selector.select_degree(TOP_K)
    energy_deg = energy_calc.compute_energy(idx_deg)
    results['Degree (Hub)'] = energy_deg
    selection_map['Degree'][idx_deg] = 1 # Mark selected

    # ==========================
    # Method 3: Random Baseline (Avg for plot, One instance for CSV)
    # ==========================
    print(f"\n[3] Running Random Selection ({RANDOM_RUNS} runs)...")
    random_energies = []
    sample_random_idx = None # Used to store a result randomly selected from 20 trials.
    
    # Randomly select one index from 20 selections (e.g., selection 0, or random selection once).
    target_random_run = np.random.randint(0, RANDOM_RUNS)
    
    for i in tqdm(range(RANDOM_RUNS)):
        idx_rnd, _ = driver_selector.select_random(TOP_K)
        e = energy_calc.compute_energy(idx_rnd)
        if e != float('inf'):
            random_energies.append(e)
        
        # Save the specified random result to the CSV tag.
        if i == target_random_run:
            sample_random_idx = idx_rnd
    
    avg_random_energy = np.mean(random_energies) if random_energies else float('inf')
    results['Random'] = avg_random_energy
    selection_map['Random'][sample_random_idx] = 1 # Mark selected

    # ==========================
    # 4. Results Summary and Export
    # ==========================
    # New feature: Save node selection status table (0/1 table)
    df_selection = pd.DataFrame(selection_map)
    selection_csv_path = os.path.join(OUTPUT_DIR, f"selection_nodes.csv")
    df_selection.to_csv(selection_csv_path, index=False)
    print(f"Node Selection Matrix saved to: {selection_csv_path}")

    # Existing function: Save energy summary CSV
    df_res = pd.DataFrame(list(results.items()), columns=['Method', 'Control_Energy_J'])
    csv_path = os.path.join(OUTPUT_DIR, f"energy_results.csv")
    df_res.to_csv(csv_path, index=False)
    
    # Drawing pictures
    plot_results(results, subgraph_name=f"{len(sub_nodes)} nodes")
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()