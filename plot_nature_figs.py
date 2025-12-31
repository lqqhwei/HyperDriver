import argparse
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
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
    "HyperDriver (Full)": "#d62728",
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

def plot_efficiency_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "efficiency_battle.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    # Data preprocessing
    main_strats = ["HyperDriver (Full)", "DC", "BC", "EC", "Random"]
    df = df[df["strategy"].isin(main_strats)].copy()
    df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"
    df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    # --- [New Feature] Export Source Data ---
    csv_out = os.path.join(root_dir, "figures", "energy_battles", f"{dataset_name}_efficiency_battle.csv")
    _ensure_parent_dir(csv_out)
    df[["selected_frac", "log_energy", "strategy"]].rename(
        columns={"selected_frac": "Fraction", "log_energy": "Log10_Energy", "strategy": "Strategy"}
    ).to_csv(csv_out, index=False)

    # Drawing Logic
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

    plt.xlabel("Fraction of Driver Nodes (Top 30%)")
    plt.ylabel("Log10(Control Energy Cost)")
    plt.title(f"Efficiency: {dataset_name}")
    plt.xlim(0.0, 0.3)
    plt.legend(frameon=True, loc="best", labelspacing=0.2, borderpad=0.2, edgecolor="gray", framealpha=0.9).get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "energy_battles", f"{dataset_name}_efficiency_battle.png"))
    plt.close()
    print(f"[SAVE] Saved Efficiency Plot & CSV for {dataset_name}")

def plot_ablation_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "efficiency_battle.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: return

    ablation_strats = ["HyperDriver (Full)", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    df = df[df["strategy"].isin(ablation_strats)].copy()
    df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"
    df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    # --- [Added] Export Source Data ---
    csv_out = os.path.join(root_dir, "figures", "ablation_battles", f"{dataset_name}_ablation_battle.csv")
    _ensure_parent_dir(csv_out)
    df[["selected_frac", "log_energy", "strategy"]].rename(
        columns={"selected_frac": "Fraction", "log_energy": "Log10_Energy", "strategy": "Strategy"}
    ).to_csv(csv_out, index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    draw_order = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    for strat in draw_order:
        sub = df[df["strategy"] == strat].sort_values("selected_frac")
        if sub.empty: continue
        color = STRATEGY_COLORS.get(strat, "black")
        lw, ls, marker, alpha, zorder = (1.2, "-", "o", 1.0, 10) if strat == "HyperDriver" else (1.0, "--", None, 0.8, 1)

        plt.plot(sub["selected_frac"], sub["log_energy"], label=strat, color=color,
                 linewidth=lw, linestyle=ls, marker=marker, markersize=3.5, alpha=alpha, zorder=zorder)

    plt.xlabel("Fraction of Driver Nodes (Top 30%)")
    plt.ylabel("Log10(Control Energy Cost)")
    plt.title(f"Ablation: {dataset_name}")
    plt.xlim(0.0, 0.3)
    plt.legend(frameon=True, loc="best", labelspacing=0.2, borderpad=0.2, edgecolor="gray", framealpha=0.9).get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "ablation_battles", f"{dataset_name}_ablation_battle.png"))
    plt.close()
    print(f"[SAVE] Saved Ablation Plot & CSV for {dataset_name}")

def plot_top_drivers_ki(root_dir: str, dataset_name: str, top_k=10):
    scores_path = os.path.join(root_dir, "results", dataset_name, "full", "node_scores.csv")
    if not os.path.exists(scores_path): return
    df = pd.read_csv(scores_path)
    if "driver_score" not in df.columns: return

    if "essential" not in df.columns:
        nodes_path = os.path.join(root_dir, "processed", dataset_name, "nodes.csv")
        if os.path.exists(nodes_path):
            nodes = pd.read_csv(nodes_path)
            df = df.merge(nodes[["protein", "essential"]], on="protein", how="left")
        if "essential" not in df.columns: df["essential"] = 0

    df = df.sort_values("driver_score", ascending=False).head(top_k)
    
    # --- [Added] Export Source Data ---
    csv_out = os.path.join(root_dir, "figures", "top_drivers", f"{dataset_name}_top_drivers.csv")
    _ensure_parent_dir(csv_out)
    df[["protein", "driver_score", "essential"]].rename(
        columns={"protein": "Protein", "driver_score": "Driver_Score", "essential": "Is_Essential"}
    ).to_csv(csv_out, index=False)

    plot_df = df.iloc[::-1] # Drawing needs to be done from bottom to top.
    proteins = plot_df["protein"].astype(str).tolist()
    vals = plot_df["driver_score"].values
    colors = ["#7f7f7f" if e == 1 else "#ff7f0e" for e in plot_df["essential"].fillna(0).values]

    fig = plt.figure(figsize=STD_FIG_SIZE)
    plt.barh(range(len(proteins)), vals, color=colors, height=0.65)
    plt.yticks(range(len(proteins)), proteins)
    plt.xlabel(r"Driver Score ($K_i$)")
    plt.title(f"Top {top_k} Drivers ({dataset_name})")
    
    legend_elements = [Patch(facecolor="#7f7f7f", label="Essential"), Patch(facecolor="#ff7f0e", label="Non-Essential")]
    plt.legend(handles=legend_elements, loc="lower left", frameon=True, borderpad=0.3, framealpha=0.5, edgecolor="gray").get_frame().set_linewidth(0.5)
    plt.tight_layout()
    
    save_publication_figure(fig, os.path.join(root_dir, "figures", "top_drivers", f"{dataset_name}_top_drivers.png"))
    plt.close()
    print(f"[SAVE] Saved Top Drivers Plot & CSV for {dataset_name}")

# ============================
# 3. Global summary plotting and data export
# ============================

def plot_global_efficiency_boxplot(df: pd.DataFrame, out_dir: str):
    if df.empty: return
    main_strats = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sub_df = df[df["Strategy"].isin(main_strats)].copy()
    
    # Export Summary CSV
    sub_df.to_csv(os.path.join(out_dir, "global_efficiency_summary.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sns.boxplot(data=sub_df, x="Strategy", y="MeanLogEnergy", order=order, palette=STRATEGY_COLORS, width=0.6, linewidth=1.0, showfliers=False)
    sns.stripplot(data=sub_df, x="Strategy", y="MeanLogEnergy", order=order, color=".3", size=3, alpha=0.6, jitter=True)
    plt.ylabel("Mean Log10 Energy Cost")
    plt.xlabel("")
    plt.title("Global Efficiency Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_efficiency_summary.png"))
    plt.close()

def plot_global_ablation_boxplot(df: pd.DataFrame, out_dir: str):
    if df.empty: return
    ablation_strats = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    sub_df = df[df["Strategy"].isin(ablation_strats)].copy()
    
    # Export Summary CSV
    sub_df.to_csv(os.path.join(out_dir, "global_ablation_summary.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    sns.boxplot(data=sub_df, x="Strategy", y="MeanLogEnergy", order=order, palette=STRATEGY_COLORS, width=0.6, linewidth=1.0, showfliers=False)
    sns.stripplot(data=sub_df, x="Strategy", y="MeanLogEnergy", order=order, color=".3", size=3, alpha=0.6, jitter=True)
    plt.ylabel("Mean Log10 Energy Cost")
    plt.xlabel("")
    plt.title("Global Ablation Summary")
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_ablation_summary.png"))
    plt.close()

def plot_global_driver_composition(df: pd.DataFrame, out_dir: str):
    if df.empty: return
    
    # Export Summary CSV
    df.to_csv(os.path.join(out_dir, "global_driver_composition.csv"), index=False)

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sns.boxplot(data=df, x="Strategy", y="EssentialFrac", order=order, palette=STRATEGY_COLORS, width=0.6, linewidth=1.0, showfliers=False)
    sns.stripplot(data=df, x="Strategy", y="EssentialFrac", order=order, color=".3", size=3, alpha=0.6, jitter=True)
    plt.ylabel("Essential Fraction (in Top-20)")
    plt.xlabel("")
    plt.title("Driver Essentiality Composition")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_publication_figure(fig, os.path.join(out_dir, "global_driver_composition.png"))
    plt.close()

# ============================
# 4. Data collection
# ============================

def collect_efficiency_metrics(root_dir: str, dataset_list: list) -> pd.DataFrame:
    records = []
    for ds in dataset_list:
        csv_path = os.path.join(root_dir, "results", ds, "full", "efficiency_battle.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"
        df = df[df["selected_frac"] <= 0.32].copy()
        df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
        grouped = df.groupby("strategy")["log_energy"].mean()
        for strat, score in grouped.items():
            records.append({"Dataset": ds, "Strategy": strat, "MeanLogEnergy": score})
    return pd.DataFrame(records)

def collect_driver_composition(root_dir: str, dataset_list: list, top_k: int = 20) -> pd.DataFrame:
    records = []
    for ds in dataset_list:
        nodes_path = os.path.join(root_dir, "processed", ds, "nodes.csv")
        if not os.path.exists(nodes_path): continue
        nodes_df = pd.read_csv(nodes_path)
        if "essential" not in nodes_df.columns: continue
        ess_labels = nodes_df["essential"].values

        hd_path = os.path.join(root_dir, "results", ds, "full", "node_scores.csv")
        if os.path.exists(hd_path):
            hd_df = pd.read_csv(hd_path)
            if "driver_score" in hd_df.columns:
                top_idx = np.argsort(-hd_df["driver_score"].values)[:top_k]
                records.append({"Dataset": ds, "Strategy": "HyperDriver", "EssentialFrac": ess_labels[top_idx].mean()})

        for method in ["dc", "bc", "ec"]:
            path = os.path.join(root_dir, "results", ds, "baselines", f"{method}_scores.csv")
            if os.path.exists(path):
                b_df = pd.read_csv(path)
                if method in b_df.columns:
                    top_idx = np.argsort(-b_df[method].values)[:top_k]
                    records.append({"Dataset": ds, "Strategy": method.upper(), "EssentialFrac": ess_labels[top_idx].mean()})
        records.append({"Dataset": ds, "Strategy": "Random", "EssentialFrac": ess_labels.mean()})
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
        plot_efficiency_battle(root_dir, ds)
        plot_ablation_battle(root_dir, ds)
        plot_top_drivers_ki(root_dir, ds, top_k=10)

    if args.dataset == "all":
        print("\n[AGGREGATING] Creating Global Summaries...")
        eff_df = collect_efficiency_metrics(root_dir, dataset_list)
        comp_df = collect_driver_composition(root_dir, dataset_list)
        out_dir = os.path.join(root_dir, "figures", "global_summary")
        os.makedirs(out_dir, exist_ok=True)
        
        plot_global_efficiency_boxplot(eff_df, out_dir)
        plot_global_ablation_boxplot(eff_df, out_dir)
        plot_global_driver_composition(comp_df, out_dir)
        print(f"[SUCCESS] Global Source Data saved in: {out_dir}")

    print("\n[DONE] All tasks completed successfully.\n")

if __name__ == "__main__":
    main()