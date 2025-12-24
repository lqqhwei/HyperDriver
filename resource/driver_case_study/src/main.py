# main.py
"""
HyperDriver Case Study Candidate Selector (Tie-break Upgrade)

核心思想不变：放弃"硬阈值交集"，采用"分层择优"，保证每个数据集都能产出三类 Case 的候选池：
1. Hidden Driver: 在 Low-Degree 群体中选 DriverScore 最高的（并列时更偏向更低 Degree、非必需）。
2. True Leader:  在 High-Degree 群体中选 DriverScore 最高的（并列时更偏向更高 Degree、非必需）。
3. Inefficient Hub: 在 High-Degree 群体中选 DriverScore 最低的（并列时更偏向更高 Degree、更低 EnergyEff、非必需）。

本版仅做两类升级（整体结构与输出文件保持不变）：
- 阈值更贴合定义：Low-Degree 默认用 bottom 30%（不足时回退到 median）；Hub 仍用 top 10%，不足时逐步放宽。
- 排序加入 tie-break：避免 DriverScore 并列（尤其 0 值）导致挑不到"最极端反差"的代表节点。

输出：
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
    从 conf/datasets.json 读取启用的数据集列表。
    只返回 enabled == true 的 name。:contentReference[oaicite:8]{index=8}
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

    # 确保列存在
    required_cols = ["protein", "driver_score", "score_S", "score_AC"]
    if not all(col in df.columns for col in required_cols):
        return []

    df = _prefer_nonessential_first(df)

    # 过滤掉孤立点 (Degree=0)，避免无意义的分析
    df = df[df["score_S"] > 0].copy()
    if df.empty:
        return []

    # 1) 定义群体阈值
    # Low-degree: bottom 30%（不足 3 个则回退到 median）
    s_low30 = get_percentile_threshold(df["score_S"], 30)
    s_median = get_percentile_threshold(df["score_S"], 50)

    # High-degree: top 10%（不足时逐步放宽到 top20/top30/top40）
    hub_percentiles = [90, 80, 70, 60]

    candidates = []

    # ---------------------------------------------------------
    # Case 1: Hidden Driver (高 DriverScore + 低 Degree)
    # ---------------------------------------------------------
    pool_non_hub = df[df["score_S"] <= s_low30].copy()
    if len(pool_non_hub) < 3:
        pool_non_hub = df[df["score_S"] <= s_median].copy()

    if not pool_non_hub.empty:
        # 排序：DriverScore ↓，Degree ↑(更小优先)，Essential ↑(0 优先)
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
    # Case 2 & 3: 在 Hub 池中择优（True Leader / Inefficient Hub）
    # ---------------------------------------------------------
    pool_hub = pd.DataFrame(columns=df.columns)
    for p in hub_percentiles:
        thr = get_percentile_threshold(df["score_S"], p)
        pool_hub = df[df["score_S"] >= thr].copy()
        if len(pool_hub) >= 3:
            break

    # 若仍不足 3，至少保证非空（极小数据集兜底）
    if pool_hub.empty:
        thr = get_percentile_threshold(df["score_S"], 50)
        pool_hub = df[df["score_S"] >= thr].copy()

    if not pool_hub.empty:
        # ---------------------------------------------------------
        # Case 2: True Leader (高 DriverScore + 高 Degree)
        # 排序：DriverScore ↓，Degree ↓(更大优先)，Essential ↑(0 优先)
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
        # Case 3: Inefficient Hub (低 DriverScore + 高 Degree)
        # 排序：DriverScore ↑(更低优先)，Degree ↓(更大优先)，EnergyEff ↑(更低优先)，Essential ↑(0 优先)
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
    从全网候选者中，选出 3 个最终代表 (The Chosen Ones)
    仅更新排序/tie-break 逻辑，输出结构保持不变。
    """
    best_picks = []

    # 1) Best Hidden Driver: DriverScore ↓, Degree ↑(更小), Essential ↑(0)
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

    # 2) Best True Leader: DriverScore ↓, Degree ↓(更大), Essential ↑(0)
    c2 = all_candidates_df[all_candidates_df["Case"] == "2_True_Leader"]
    if not c2.empty:
        best_c2 = c2.sort_values(
            by=["DriverScore", "Degree", "Essential"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        best_picks.append(best_c2)

    # 3) Best Inefficient Hub: DriverScore ↑(更低), Degree ↓(更大), EnergyEff ↑(更低), Essential ↑(0)
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

    # 1) 保存候选池
    os.makedirs(os.path.join(root_dir, "resource/driver_case_study/output"), exist_ok=True)
    out_path = os.path.join(root_dir, "resource/driver_case_study/output", "candidates.csv")
    res_df.to_csv(out_path, index=False)
    print(f"[INFO] Full candidates saved to: {out_path}")

    # 2) 决出 Top 3
    print("\n========== The Final Three ==========")
    best_df = select_best_representatives(res_df)

    best_out_path = os.path.join(root_dir, "resource/driver_case_study/output", "driver_results.csv")
    best_df.to_csv(best_out_path, index=False)

    # 打印到控制台
    print(best_df.to_string(index=False))
    print(f"\n[SUCCESS] Final representatives saved to: {best_out_path}")


if __name__ == "__main__":
    main()
