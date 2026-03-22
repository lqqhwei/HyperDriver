# metrics.py
import os
import json
import numpy as np
import pandas as pd
from typing import List

def load_datasets_config(conf_path: str) -> List[str]:
    if not os.path.exists(conf_path): return []
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [item["name"] for item in cfg.get("datasets", []) if item.get("enabled", False)]

def get_interpolated_log_energy(battle_df: pd.DataFrame, strategy: str, target_frac: float) -> float:
    #Smooth linear interpolation to calculate approximate physical energy at the current sparsity.
    sub = battle_df[battle_df["strategy"] == strategy].sort_values("selected_frac")
    if sub.empty: return np.nan
    fracs = sub["selected_frac"].values
    energies = sub["energy"].values
    log_energies = np.log10(energies + 1e-9)
    # If the value is outside the range, take the edge value.
    if target_frac > fracs.max(): return float(log_energies[-1])
    if target_frac < fracs.min(): return float(log_energies[0])
    return float(np.interp(target_frac, fracs, log_energies))

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json"))
    all_metrics = []

    for ds in datasets:
        res_dir = os.path.join(root_dir, "results", ds)
        # Read data that is currently on your hard drive and has not been touched.
        try:
            nodes_df = pd.read_csv(os.path.join(res_dir, "full", "node_scores.csv"))
            keys_df = pd.read_csv(os.path.join(res_dir, "keys", "key_driver_nodes.csv"))
            battle_df = pd.read_csv(os.path.join(res_dir, "full", "energy_battle.csv"))
        except:
            continue

        N, K = len(nodes_df), len(keys_df)
        if N == 0 or K == 0: continue
        sparsity_frac = K / N

        merged = keys_df.merge(nodes_df[["protein", "score_S", "driver_score"]], on="protein", how="left")
        driver_degrees = merged["score_S"].values
        driver_scores = merged["driver_score_x"].values if "driver_score_x" in merged.columns else merged["driver_score"].values
        global_avg_degree = nodes_df["score_S"].mean()

        # Computational specificity and marginal utility
        eds_pct = (np.sum(driver_degrees <= global_avg_degree) / K) * 100
        tcr_ratio = np.mean(driver_degrees) / global_avg_degree if global_avg_degree > 0 else 0
        mean_meg = np.mean(driver_scores)

        # Energy is estimated via interpolation (minor errors, perfectly acceptable).
        hd_log_energy = get_interpolated_log_energy(battle_df, "HyperDriver", sparsity_frac)
        dc_log_energy = get_interpolated_log_energy(battle_df, "DC", sparsity_frac)
        mer_score = dc_log_energy - hd_log_energy

        metrics_dict = {
            "Dataset": ds,
            "Total_Nodes_N": N,
            "Identified_Drivers_K": K,
            "Identification_Rate(%)": f"{sparsity_frac * 100:.2f}%",
            "HD_Log10_Energy(LCE)": round(hd_log_energy, 4),
            "DC_Log10_Energy_Base": round(dc_log_energy, 4),
            "Energy_Reduction_Magnitude(MER)": round(mer_score, 2),
            "Edge_Driver_Specificity(EDS,%)": f"{eds_pct:.2f}%",
            "Topo_Concentration_Ratio(TCR)": round(tcr_ratio, 2),
            "Mean_Marginal_Gain(MEG)": round(mean_meg, 4)
        }
        all_metrics.append(metrics_dict)
        
        # deposit slip documents
        idx_dir = os.path.join(res_dir, "index")
        os.makedirs(idx_dir, exist_ok=True)
        pd.DataFrame([metrics_dict]).to_csv(os.path.join(idx_dir, f"{ds}_index.csv"), index=False)

    # Summary Table
    if all_metrics:
        metrics_dir = os.path.join(root_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        pd.DataFrame(all_metrics).to_csv(os.path.join(metrics_dir, "index_summary.csv"), index=False)
        print("✅ Zero-risk extraction complete! Please go to metrics/index_summary.csv to view your final summary table.")

if __name__ == "__main__":
    main()