# main.py
"""
HyperDriver Case Study Candidate Selector (Tie-break Upgrade)

The core idea remains unchanged: abandoning the "hard threshold intersection" approach, a "stratified selection" method is adopted to ensure that each dataset produces candidate pools for three types of cases:
1. Hidden Driver: Select the driver with the highest DriverScore from the Low-Degree group (in case of a tie, favoring lower degrees; not mandatory).
2. True Leader: Select the driver with the highest DriverScore from the High-Degree group (in case of a tie, favoring higher degrees; not mandatory).
3. Inefficient Hub: Select the hub with the lowest DriverScore from the High-Degree group (in case of a tie, favoring higher degrees and lower EnergyEff; not mandatory).

This version only upgrades two categories (the overall structure and output files remain unchanged):
- Thresholds are more closely aligned with the definition: Low-Degree uses the bottom 30% by default (falling back to the median if insufficient); Hub still uses the top 10%, gradually widening if insufficient.
- Added tie-break to sorting: This prevents DriverScore ties (especially 0 values) from failing to select the representative node with the "most extreme contrast".

Output:
- output/candidates.csv
- output/driver_results.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from typing import List
from pathlib import Path

def load_datasets_config(conf_path: str) -> List[str]:
    """
    Read the list of enabled datasets from conf/datasets.json.
    Returns only the name where enabled == true. :contentReference[oaicite:8]{index=8}
    """
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    enabled = []
    for item in cfg.get("datasets", []):
        if item.get("enabled", False):
            enabled.append(item["name"])
    return enabled

def get_percentile_threshold(series, percentile):
    """Return the value at given percentile for a numeric series."""
    return np.percentile(series, percentile)


def _prefer_nonessential_first(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an 'essential' column exists and is 0/1 numeric for sorting."""
    if "essential" not in df.columns:
        df["essential"] = 0
    df["essential"] = pd.to_numeric(df["essential"], errors="coerce").fillna(0).astype(int)
    return df


def find_candidates_for_dataset(root_dir, dataset_name):
    scores_path = os.path.join(root_dir, "results", dataset_name, "full", "node_scores.csv")
    if not os.path.exists(scores_path):
        return []

    df = pd.read_csv(scores_path)

    # Ensure the column exists
    required_cols = ["protein", "driver_score", "score_S", "score_AC"]
    if not all(col in df.columns for col in required_cols):
        return []

    df = _prefer_nonessential_first(df)

    # Filter out outliers (Degree=0) to avoid meaningless analysis.
    df = df[df["score_S"] > 0].copy()
    if df.empty:
        return []

    # 1) Define group threshold
    # Low-degree: bottom 30% (if less than 3, it will fall back to the median).
    s_low30 = get_percentile_threshold(df["score_S"], 30)
    s_median = get_percentile_threshold(df["score_S"], 50)

    # High-degree: Top 10% (if this is not met, the requirement will be gradually relaxed to Top 20/Top 30/Top 40)
    hub_percentiles = [90, 80, 70, 60]

    candidates = []

    # ---------------------------------------------------------
    # Case 1: Hidden Driver (High DriverScore + Low Degree)
    # ---------------------------------------------------------
    pool_non_hub = df[df["score_S"] <= s_low30].copy()
    if len(pool_non_hub) < 3:
        pool_non_hub = df[df["score_S"] <= s_median].copy()

    if not pool_non_hub.empty:
        # Sorting: DriverScore ↓, Degree ↑ (lower priority), Essential ↑ (0 priority)
        top_hidden = pool_non_hub.sort_values(
            by=["driver_score", "score_S", "essential"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(3)

        for _, row in top_hidden.iterrows():
            candidates.append(
                {
                    "Dataset": dataset_name,
                    "Case": "1_Hidden_Driver",
                    "Protein": row["protein"],
                    "DriverScore": row["driver_score"],
                    "Degree": row["score_S"],
                    "EnergyEff": row["score_AC"],
                    "Essential": int(row.get("essential", 0)),
                }
            )

    # ---------------------------------------------------------
    # Cases 2 & 3: Selecting the best leader from the hub pool (True Leader / Inefficient Hub)
    # ---------------------------------------------------------
    pool_hub = pd.DataFrame(columns=df.columns)
    for p in hub_percentiles:
        thr = get_percentile_threshold(df["score_S"], p)
        pool_hub = df[df["score_S"] >= thr].copy()
        if len(pool_hub) >= 3:
            break

    # If the result is still less than 3, at least ensure that it is not empty (for very small datasets as a safety net).
    if pool_hub.empty:
        thr = get_percentile_threshold(df["score_S"], 50)
        pool_hub = df[df["score_S"] >= thr].copy()

    if not pool_hub.empty:
        # ---------------------------------------------------------
        # Case 2: True Leader (High DriverScore + High Degree)
        # Sorting: DriverScore ↓, Degree ↓ (higher priority), Essential ↑ (0 priority)
        # ---------------------------------------------------------
        top_leader = pool_hub.sort_values(
            by=["driver_score", "score_S", "essential"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(3)

        for _, row in top_leader.iterrows():
            candidates.append(
                {
                    "Dataset": dataset_name,
                    "Case": "2_True_Leader",
                    "Protein": row["protein"],
                    "DriverScore": row["driver_score"],
                    "Degree": row["score_S"],
                    "EnergyEff": row["score_AC"],
                    "Essential": int(row.get("essential", 0)),
                }
            )

        # ---------------------------------------------------------
        # Case 3: Inefficient Hub (low DriverScore + high Degree)
        # Sorting: DriverScore ↑ (lower priority), Degree ↓ (higher priority), EnergyEff ↑ (lower priority), Essential ↑ (0 priority)
        # ---------------------------------------------------------
        bad_hub = pool_hub.sort_values(
            by=["driver_score", "score_S", "score_AC", "essential"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).head(3)

        for _, row in bad_hub.iterrows():
            candidates.append(
                {
                    "Dataset": dataset_name,
                    "Case": "3_Inefficient_Hub",
                    "Protein": row["protein"],
                    "DriverScore": row["driver_score"],
                    "Degree": row["score_S"],
                    "EnergyEff": row["score_AC"],
                    "Essential": int(row.get("essential", 0)),
                }
            )

    return candidates


def select_best_representatives(all_candidates_df):
    """
    Three final representatives were selected from all online candidates (The Chosen Ones).
    Only the sorting/tie-break logic is updated; the output structure remains unchanged.
    """
    best_picks = []

    # 1) Best Hidden Driver: DriverScore ↓, Degree ↑(smaller), Essential ↑(0)
    c1 = all_candidates_df[all_candidates_df["Case"] == "1_Hidden_Driver"]
    if not c1.empty:
        best_c1 = c1.sort_values(
            by=["DriverScore", "Degree", "Essential"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        best_picks.append(best_c1)
    else:
        best_picks.append(pd.Series({"Case": "1_Hidden_Driver", "Protein": "None"}))

    # 2) Best True Leader: DriverScore ↓, Degree ↓(greater), Essential ↑(0)
    c2 = all_candidates_df[all_candidates_df["Case"] == "2_True_Leader"]
    if not c2.empty:
        best_c2 = c2.sort_values(
            by=["DriverScore", "Degree", "Essential"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        best_picks.append(best_c2)

    # 3) Best Inefficient Hub: DriverScore ↑ (Lower), Degree ↓ (Higher), EnergyEff ↑ (Lower), Essential ↑ (0)
    c3 = all_candidates_df[all_candidates_df["Case"] == "3_Inefficient_Hub"]
    if not c3.empty:
        best_c3 = c3.sort_values(
            by=["DriverScore", "Degree", "EnergyEff", "Essential"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).iloc[0]
        best_picks.append(best_c3)

    return pd.DataFrame(best_picks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[3]
    
    if args.dataset == "all":
        dataset_list = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json"))
    else:
        dataset_list = [args.dataset]

    all_candidates = []
    print("========== Screening Candidates (V3.1 Tie-break Upgrade) ==========")

    for dataset_name in dataset_list:
        cands = find_candidates_for_dataset(root_dir, dataset_name)
        if cands:
            all_candidates.extend(cands)
            print(f"[OK] {dataset_name}: {len(cands)} candidates")
        else:
            print(f"[SKIP] {dataset_name}: no candidates or missing files")

    if not all_candidates:
        print("[ERROR] No candidates found. Please check input files and paths.")
        return

    res_df = pd.DataFrame(all_candidates)

    # 1) Save candidate pool
    os.makedirs(os.path.join(root_dir, "resource/driver_case_study/output"), exist_ok=True)
    out_path = os.path.join(root_dir, "resource/driver_case_study/output", "candidates.csv")
    res_df.to_csv(out_path, index=False)
    print(f"[INFO] Full candidates saved to: {out_path}")

    # 2) Top 3 determined
    print("\n========== The Final Three ==========")
    best_df = select_best_representatives(res_df)

    best_out_path = os.path.join(root_dir, "resource/driver_case_study/output", "driver_results.csv")
    best_df.to_csv(best_out_path, index=False)

    # Print to console
    print(best_df.to_string(index=False))
    print(f"\n[SUCCESS] Final representatives saved to: {best_out_path}")


if __name__ == "__main__":
    main()
