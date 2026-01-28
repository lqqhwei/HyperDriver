import argparse
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib

warnings.filterwarnings("ignore")
from src.data_utils import load_datasets_config

# ============================
# 1. Unified drawing style settings
# ============================
STD_FIG_SIZE = (3.5, 3.0)

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.transparent"] = False

# Font size steps (optimized for 3.5x3.0 inch canvas)
plt.rcParams["font.size"] = 7
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

# Line and layout precision
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["lines.markersize"] = 3.5
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600

STRATEGY_COLORS = {
    "HyperDriver": "#d62728",
    "HyperDriver": "#d62728",
    "DC": "#ff7f0e",
    "BC": "#2ca02c",
    "EC": "#9467bd",
    "Random": "#7f7f7f",
    "w/o Greedy": "#8c564b",
    "w/o Hypergraph": "#bcbd22",
    "w/o Dynamics": "#17becf",
}

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def save_publication_figure(fig, out_path: str) -> None:
    """Save as PDF + PNG and ensure a physical white background."""
    _ensure_parent_dir(out_path)
    base_path, _ = os.path.splitext(out_path)
    try:
        fig.patch.set_facecolor("white")
    except Exception: pass

    fig.savefig(f"{base_path}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02, transparent=False)
    fig.savefig(f"{base_path}.png", format="png", bbox_inches="tight", pad_inches=0.02, transparent=False)

# ============================
# 2. Plotting and exporting data from a single dataset
# ============================

def plot_energy_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "energy_battle.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    main_strats = ["HyperDriver", "DC", "BC", "EC", "Random"]
    df = df[df["strategy"].isin(main_strats)].copy()
    df.loc[df["strategy"] == "HyperDriver", "strategy"] = "HyperDriver"
    df["log_energy"] = np.log10(df["energy"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    csv_out = os.path.join(root_dir, "figures", "energy_battles", f"{dataset_name}_energy_battle.csv")
    _ensure_parent_dir(csv_out)
    df[["selected_frac", "log_energy", "strategy"]].rename(
        columns={"selected_frac": "Fraction", "log_energy": "LogEnergy", "strategy": "Strategy"}
    ).to_csv(csv_out, index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    draw_order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    strategies = df["strategy"].unique()

    for strat in draw_order:
        if strat not in strategies: continue
        sub = df[df["strategy"] == strat].sort_values("selected_frac")
        color = STRATEGY_COLORS.get(strat, "black")
        lw = 1.2 if strat == "HyperDriver" else 1.0
        ls = "-" if strat == "HyperDriver" else ("--" if strat == "Random" else "-.")
        marker = "o" if strat == "HyperDriver" else None
        zorder = 10 if strat == "HyperDriver" else 1

        plt.plot(sub["selected_frac"], sub["log_energy"], label=strat, color=color, 
                 linewidth=lw, linestyle=ls, marker=marker, markersize=3.5, zorder=zorder)

    plt.xlabel("Fraction of Driver Nodes(Top 30%)")
    plt.ylabel("Log10(Control Energy)")
    plt.title(f"Control Energy({dataset_name})")
    plt.xlim(0.0, 0.3)
    plt.legend(frameon=True, loc="best", labelspacing=0.2, borderpad=0.2, edgecolor="gray", framealpha=0.9).get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "energy_battles", f"{dataset_name}_energy_battle.png"))
    plt.close()
    print(f"[SAVE] Saved Energy Plot & CSV for {dataset_name}")

def plot_ablation_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "energy_battle.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    ablation_strats = ["HyperDriver", "w/o Dynamics", "w/o Hypergraph", "w/o Greedy"]
    df = df[df["strategy"].isin(ablation_strats)].copy()
    df.loc[df["strategy"] == "HyperDriver", "strategy"] = "HyperDriver"
    df["log_energy"] = np.log10(df["energy"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    csv_out = os.path.join(root_dir, "figures", "ablation_battles", f"{dataset_name}_ablation_battle.csv")
    _ensure_parent_dir(csv_out)
    df[["selected_frac", "log_energy", "strategy"]].rename(
        columns={"selected_frac": "Fraction", "log_energy": "LogEnergy", "strategy": "Strategy"}
    ).to_csv(csv_out, index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    draw_order = ["HyperDriver", "w/o Dynamics", "w/o Hypergraph", "w/o Greedy"]
    for strat in draw_order:
        sub = df[df["strategy"] == strat].sort_values("selected_frac")
        if sub.empty: continue
        color = STRATEGY_COLORS.get(strat, "black")
        lw, ls, marker, alpha, zorder = (1.2, "-", "o", 1.0, 10) if strat == "HyperDriver" else (1.0, "--", None, 0.8, 1)

        plt.plot(sub["selected_frac"], sub["log_energy"], label=strat, color=color,
                 linewidth=lw, linestyle=ls, marker=marker, markersize=3.5, alpha=alpha, zorder=zorder)

    plt.xlabel("Fraction of Driver Nodes(Top 30%)")
    plt.ylabel("Log10(Control Energy)")
    plt.title(f"Ablation({dataset_name})")
    plt.xlim(0.0, 0.3)
    
    # [FIXED] Force legend to upper right based on user feedback
    plt.legend(frameon=True, loc="upper right", labelspacing=0.2, borderpad=0.2, edgecolor="gray", framealpha=0.9).get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "ablation_battles", f"{dataset_name}_ablation_battle.png"))
    plt.close()
    print(f"[SAVE] Saved Ablation Plot & CSV for {dataset_name}")

def plot_top_drivers_ki(root_dir: str, dataset_name: str, top_k=10):
    scores_path = os.path.join(root_dir, "results", dataset_name, "full", "node_scores.csv")
    if not os.path.exists(scores_path): return
    df = pd.read_csv(scores_path)
    if "driver_score" not in df.columns: return

    # Ensure degree score exists
    if "score_S" not in df.columns: df["score_S"] = 0

    # [MODIFIED] No longer merging essential labels for visualization purposes
    # The chart will just show Driver Score vs Degree, agnostic of essentiality.

    df = df.sort_values("driver_score", ascending=False).head(top_k)
    
    # --- Export Source Data ---
    csv_out = os.path.join(root_dir, "figures", "top_drivers", f"{dataset_name}_top_drivers.csv")
    _ensure_parent_dir(csv_out)
    # Note: Export data might still want to keep essential info for reference if available, 
    # but for visualization we drop it. Let's keep data clean.
    export_cols = ["protein", "driver_score", "score_S"]
    rename_map = {"protein": "Protein", "driver_score": "Driver_Score", "score_S": "Degree_Score"}
    df[export_cols].rename(columns=rename_map).to_csv(csv_out, index=False)

    # Plotting Data
    proteins = df["protein"].astype(str).tolist()
    driver_scores = df["driver_score"].values
    degrees = df["score_S"].values
    
    # [MODIFIED] Uniform color for all bars (Orange = #ff7f0e)
    colors = ["#ff7f0e"] * len(proteins)

    fig, ax1 = plt.subplots(figsize=STD_FIG_SIZE)
    
    # Left Axis: Driver Score (Bar)
    x_pos = range(len(proteins))
    ax1.bar(x_pos, driver_scores, color=colors, width=0.65, label="Driver Score", zorder=2)
    ax1.set_ylabel(r"Driver Score($K_i$)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(proteins, rotation=45, ha='right')
    ax1.grid(axis='x', which='both', alpha=0) 
    
    # Right Axis: Degree (Line)
    ax2 = ax1.twinx()
    ax2.plot(x_pos, degrees, color="#1f77b4", marker="o", markersize=4, linewidth=1.5, label="Degree Score", zorder=3)
    ax2.set_ylabel("Degree Score")
    ax2.grid(False) 

    plt.title(f"Top{top_k} Drivers({dataset_name})", pad=15)
    
    # [MODIFIED] Simplified Legend: Driver Score (Orange) & Degree (Blue)
    legend_elements = [
        Patch(facecolor="#ff7f0e", label="Driver Score"), 
        Line2D([0], [0], color="#1f77b4", marker='o', lw=1.5, label='Degree Score')
    ]
    
    ax1.legend(
        handles=legend_elements, 
        loc="lower center", 
        bbox_to_anchor=(0.5, 0.98), 
        ncol=2,               
        frameon=False,        
        borderaxespad=0.5,    
        fontsize=6,           
        handletextpad=0.4,    
        columnspacing=1.0     
    )
    
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "top_drivers", f"{dataset_name}_top_drivers.png"))
    plt.close()
    print(f"[SAVE] Saved Top Drivers Dual-Axis Plot & CSV for {dataset_name}")

# ============================
# 3. Global summary plotting and data export
# ============================

def plot_global_efficiency_boxplot(df: pd.DataFrame, out_dir: str):
    if df.empty: return
    main_strats = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sub_df = df[df["Strategy"].isin(main_strats)].copy()
    
    sub_df.to_csv(os.path.join(out_dir, "global_energy_summary.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sns.boxplot(data=sub_df, x="Strategy", y="LogEnergy", order=order, palette=STRATEGY_COLORS, width=0.6, linewidth=1.0, showfliers=False)
    sns.stripplot(data=sub_df, x="Strategy", y="LogEnergy", order=order, color=".3", size=3, alpha=0.6, jitter=True)
    plt.ylabel("Log10(Control Energy)")
    plt.xlabel("")
    plt.title("Control Energy(Global)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_energy_summary.png"))
    plt.close()

def plot_global_ablation_boxplot(df: pd.DataFrame, out_dir: str):
    if df.empty: return
    ablation_strats = ["HyperDriver", "w/o Dynamics", "w/o Hypergraph", "w/o Greedy"]
    sub_df = df[df["Strategy"].isin(ablation_strats)].copy()
    
    sub_df.to_csv(os.path.join(out_dir, "global_ablation_summary.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "w/o Dynamics", "w/o Hypergraph", "w/o Greedy"]
    sns.boxplot(data=sub_df, x="Strategy", y="LogEnergy", order=order, palette=STRATEGY_COLORS, width=0.6, linewidth=1.0, showfliers=False)
    sns.stripplot(data=sub_df, x="Strategy", y="LogEnergy", order=order, color=".3", size=3, alpha=0.6, jitter=True)
    plt.ylabel("Log10(Control Energy)")
    plt.xlabel("")
    plt.title("Ablation(Global)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_ablation_summary.png"))
    plt.close()

def plot_global_driver_composition(df: pd.DataFrame, out_dir: str):
    # [MODIFIED] Only Degree Fraction remains
    if df.empty: return
    
    df.to_csv(os.path.join(out_dir, "global_driver_summary.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    
    # [MODIFIED] Standard Boxplot: Single hue (by strategy color), standard width
    # We use y="DegreeFrac" directly.
    sns.boxplot(
        data=df, 
        x="Strategy", 
        y="DegreeFrac", 
        order=order, 
        palette=STRATEGY_COLORS, 
        width=0.6, 
        linewidth=1.0, 
        showfliers=False
    )
    
    # Optional: Add stripplot for detail
    sns.stripplot(data=df, x="Strategy", y="DegreeFrac", order=order, color=".3", size=3, alpha=0.6, jitter=True)

    # Axis Labels
    # [MODIFIED] Label changed to Degree Fraction
    plt.ylabel("Degree Fraction(in Top 10%)")
    plt.xlabel("")
    plt.title("Degree Fraction(Global)")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    
    # [MODIFIED] No Legend needed as X-axis labels are sufficient and self-explanatory
    
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_driver_summary.png"))
    plt.close()

# ============================
# 4. Data collection
# ============================

def collect_efficiency_metrics(root_dir: str, dataset_list: list) -> pd.DataFrame:
    records = []
    for ds in dataset_list:
        csv_path = os.path.join(root_dir, "results", ds, "full", "energy_battle.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        df.loc[df["strategy"] == "HyperDriver", "strategy"] = "HyperDriver"
        df = df[df["selected_frac"] <= 0.32].copy()
        df["log_energy"] = np.log10(df["energy"] + 1e-9)
        grouped = df.groupby("strategy")["log_energy"].mean()
        for strat, score in grouped.items():
            records.append({"Dataset": ds, "Strategy": strat, "LogEnergy": score})
    return pd.DataFrame(records)

def collect_driver_composition(root_dir: str, dataset_list: list, fraction: float = 0.10) -> pd.DataFrame:
    """
    Collects ONLY Degree Fraction for Top X% nodes.
    [MODIFIED] Removed EssentialFrac calculation.
    """
    records = []
    for ds in dataset_list:
        # 1. Load Nodes (Used only for N count now)
        nodes_path = os.path.join(root_dir, "processed", ds, "nodes.csv")
        if not os.path.exists(nodes_path): continue
        nodes_df = pd.read_csv(nodes_path)
        
        N = len(nodes_df)
        top_k = int(max(1, N * fraction))
        
        # 2. Load/Calculate Degree for ALL nodes
        hd_path = os.path.join(root_dir, "results", ds, "full", "node_scores.csv")
        all_degrees = None
        
        if os.path.exists(hd_path):
            hd_df = pd.read_csv(hd_path)
            if "score_S" in hd_df.columns:
                all_degrees = hd_df["score_S"].values
        
        if all_degrees is None:
            static_path = os.path.join(root_dir, "processed", ds, "static_edges.csv")
            if os.path.exists(static_path):
                s_df = pd.read_csv(static_path)
                deg_map = np.zeros(N)
                for _, row in s_df.iterrows():
                    u, v = int(row['src_idx']), int(row['dst_idx'])
                    if u < N: deg_map[u] += 1
                    if v < N: deg_map[v] += 1
                all_degrees = deg_map
            else:
                all_degrees = np.zeros(N)

        total_degree_sum = np.sum(all_degrees) + 1e-9

        # Helper to compute metrics (Degree Only)
        def calc_metrics(scores, strategy_name):
            top_idx = np.argsort(-scores)[:top_k]
            deg_frac = np.sum(all_degrees[top_idx]) / total_degree_sum
            return {
                "Dataset": ds, 
                "Strategy": strategy_name, 
                "DegreeFrac": deg_frac
            }

        # 3. Strategy: HyperDriver
        if os.path.exists(hd_path):
            hd_df = pd.read_csv(hd_path)
            if "driver_score" in hd_df.columns:
                records.append(calc_metrics(hd_df["driver_score"].values, "HyperDriver"))

        # 4. Strategy: Baselines
        for method in ["dc", "bc", "ec"]:
            path = os.path.join(root_dir, "results", ds, "baselines", f"{method}_scores.csv")
            if os.path.exists(path):
                b_df = pd.read_csv(path)
                if method in b_df.columns:
                    records.append(calc_metrics(b_df[method].values, method.upper()))
        
        # 5. Strategy: Random
        rand_idx = np.random.choice(N, top_k, replace=False)
        rand_deg = np.sum(all_degrees[rand_idx]) / total_degree_sum
        
        records.append({
            "Dataset": ds, 
            "Strategy": "Random", 
            "DegreeFrac": rand_deg
        })

    return pd.DataFrame(records)

# ============================
# 5. Main program entry point
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_list = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json")) if args.dataset == "all" else [args.dataset]

    print("\n========== Generating Figures & Source Data ==========")
    for ds in dataset_list:
        plot_energy_battle(root_dir, ds)
        plot_ablation_battle(root_dir, ds)
        plot_top_drivers_ki(root_dir, ds, top_k=10)

    if args.dataset == "all":
        print("\n[AGGREGATING] Creating Global Summaries...")
        eff_df = collect_efficiency_metrics(root_dir, dataset_list)
        comp_df = collect_driver_composition(root_dir, dataset_list, fraction=0.10)
        out_dir = os.path.join(root_dir, "figures", "global_summary")
        os.makedirs(out_dir, exist_ok=True)
        
        plot_global_efficiency_boxplot(eff_df, out_dir)
        plot_global_ablation_boxplot(eff_df, out_dir)
        plot_global_driver_composition(comp_df, out_dir)
        print(f"[SUCCESS] Global Source Data saved in: {out_dir}")

    print("\n[DONE] All tasks completed successfully.\n")

if __name__ == "__main__":
    main()