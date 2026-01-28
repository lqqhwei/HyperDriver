import os
import argparse
import pandas as pd
import numpy as np
import json
from typing import List
from pathlib import Path

# =============================================================================
# Case Study Selector (TOPSIS Geometric Logic - Clean)
#
# Methodology:
#   Using Euclidean Distance to "Ideal Points" in a normalized 2D feature space
#   (Log-Degree vs. Driver_Score) to identify the most extreme topological-control
#   decoupling examples.
#
#   * Essential column is REMOVED to focus purely on network physics.
#
# Ideal Points:
#   1. Peripheral_Hub:    (Score=1.0, Normalized_Degree=0.0)
#   2. High_Efficiency_Hub:      (Score=1.0, Normalized_Degree=1.0)
#   3. Low_Efficiency_Hub:  (Score=0.0, Normalized_Degree=1.0)
# =============================================================================

def load_datasets_config(conf_path: str) -> List[str]:
    """Read enabled datasets from config."""
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [item["name"] for item in cfg.get("datasets", []) if item.get("enabled", False)]

def load_all_nodes(root_dir, dataset_list) -> pd.DataFrame:
    """
    Load node scores from all datasets into a single global pool.
    Ignores 'EnergyEff' and 'Essential' columns.
    """
    all_data = []
    print(f"Loading data from {len(dataset_list)} datasets...")
    
    for dataset_name in dataset_list:
        path = os.path.join(root_dir, "results", dataset_name, "full", "node_scores.csv")
        if not os.path.exists(path):
            continue
        
        df = pd.read_csv(path)
        
        # Standardize column names
        rename_map = {
            "protein": "Protein",
            "driver_score": "Driver_Score",
            "score_S": "Degree_Score"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Add Dataset tag
        df["Dataset"] = dataset_name
        
        # Only keep strictly necessary columns for 
        cols = ["Dataset", "Protein", "Driver_Score", "Degree_Score"]
        
        # Robust selection (only if columns exist)
        final_cols = [c for c in cols if c in df.columns]
        all_data.append(df[final_cols])
        
    if not all_data:
        return pd.DataFrame()
        
    giant_df = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Total nodes loaded: {len(giant_df)}")
    return giant_df

def select_representatives_topsis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geometric Distance Logic (TOPSIS-like).
    """
    results = []
    
    # ---------------------------------------------------------
    # Step 1: Global Normalization
    # ---------------------------------------------------------
    # Handle Degree Power Law with Log10, then Min-Max Scale to [0, 1]
    
    # Avoid log(0) by adding 1 (Degree is usually >=1, but safety first)
    log_degree = np.log10(df["Degree_Score"] + 1)
    
    min_log = log_degree.min()
    max_log = log_degree.max()
    
    # Norm_Degree: 0.0 = Smallest Node in all datasets, 1.0 = Biggest Hub
    df["Norm_Degree"] = (log_degree - min_log) / (max_log - min_log)
    
    # Driver_Score is assumed to be already [0, 1]
    
    # ---------------------------------------------------------
    # Case 1: Peripheral_Hub
    # Target: Score=1.0 (Max Control), Degree=0.0 (Min Topology)
    # ---------------------------------------------------------
    print("Calculating Case 1 (Peripheral_Hub)...")
    
    # Euclidean Distance formula
    df["Dist_C1"] = np.sqrt( (df["Driver_Score"] - 1.0)**2 + (df["Norm_Degree"] - 0.0)**2 )
    
    # Sort by Distance (Ascending) -> Closest to Ideal wins
    # Tie-break with Protein name
    c1_best = df.sort_values(by=["Dist_C1", "Protein"], ascending=[True, True]).iloc[0]
    
    c1_entry = c1_best.copy()
    c1_entry["Case"] = "Peripheral_Hub"
    results.append(c1_entry)

    # ---------------------------------------------------------
    # Case 2: High_Efficiency_Hub
    # Target: Score=1.0 (Max Control), Degree=1.0 (Max Topology)
    # ---------------------------------------------------------
    print("Calculating Case 2 (True Leader)...")
    
    df["Dist_C2"] = np.sqrt( (df["Driver_Score"] - 1.0)**2 + (df["Norm_Degree"] - 1.0)**2 )
    
    c2_best = df.sort_values(by=["Dist_C2", "Protein"], ascending=[True, True]).iloc[0]
    
    c2_entry = c2_best.copy()
    c2_entry["Case"] = "High_Efficiency_Hub"
    results.append(c2_entry)

    # ---------------------------------------------------------
    # Case 3: Low_Efficiency_Hub
    # Target: Score=0.0 (Min Control), Degree=1.0 (Max Topology)
    # ---------------------------------------------------------
    print("Calculating Case 3 (Low_Efficiency_Hub)...")
    
    df["Dist_C3"] = np.sqrt( (df["Driver_Score"] - 0.0)**2 + (df["Norm_Degree"] - 1.0)**2 )
    
    c3_best = df.sort_values(by=["Dist_C3", "Protein"], ascending=[True, True]).iloc[0]
    
    c3_entry = c3_best.copy()
    c3_entry["Case"] = "Low_Efficiency_Hub"
    results.append(c3_entry)

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[3]
    
    # 1. Config Loading
    if args.dataset == "all":
        dataset_list = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json"))
    else:
        dataset_list = [args.dataset]

    # 2. Load Data
    giant_df = load_all_nodes(root_dir, dataset_list)
    
    if giant_df.empty:
        print("[ERROR] No data loaded. Check paths and enabled datasets.")
        return

    # 3. Apply Logic
    best_df = select_representatives_topsis(giant_df)
    
    # 4. Final Clean Output
    cols_order = ["Dataset", "Case", "Protein", "Driver_Score", "Degree_Score"]
    best_df = best_df[cols_order]

    # 5. Save
    os.makedirs(os.path.join(root_dir, "resource/driver_case_study/output"), exist_ok=True)
    out_path = os.path.join(root_dir, "resource/driver_case_study/output", "driver_results.csv")
    best_df.to_csv(out_path, index=False)

    print("\n========== FINAL RESULTS (Clean TOPSIS) ==========")
    print(best_df.to_string(index=False))
    print(f"\n[SUCCESS] Saved to: {out_path}")

if __name__ == "__main__":
    main()