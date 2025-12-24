# plot_nature_figs.py

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
# 1. 绘图风格全局统一设置 (SCI Standard)
# ============================
# 定义标准单栏尺寸: 3.5 英寸宽 x 3.0 英寸高
STD_FIG_SIZE = (3.5, 3.0)

# 字体家族 (Type 42 保证 PDF 矢量文字可编辑)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 数学字体尽量与正文一致（避免“正文是 Arial，数学却像 CM”的割裂感）
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

# 启用 Seaborn 白格样式作为底板 (Whitegrid 默认是有全边框的，只要不 Despine)
plt.style.use("seaborn-v0_8-whitegrid")

# 强制白底（避免 PDF/PNG 在某些环境里出现黑底/黑边条）
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.transparent"] = False

# 字体大小阶梯 (针对 3.5x3.0 英寸画布优化)
plt.rcParams["font.size"] = 7            # 全局基础字号
plt.rcParams["axes.labelsize"] = 8       # 轴标签 (X/Y Label) - 标准大小
plt.rcParams["axes.titlesize"] = 9       # 标题 (Title) - 略大
plt.rcParams["legend.fontsize"] = 7      # 图例 (Legend) - 紧凑
plt.rcParams["xtick.labelsize"] = 7      # X轴刻度
plt.rcParams["ytick.labelsize"] = 7      # Y轴刻度

# 线条与布局精度
plt.rcParams["axes.linewidth"] = 0.8     # 坐标轴线宽 (四周统一)
plt.rcParams["grid.linewidth"] = 0.5     # 网格线宽
plt.rcParams["xtick.major.width"] = 0.8  # 刻度线宽
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["lines.linewidth"] = 1.2    # 折线图线宽
plt.rcParams["lines.markersize"] = 3.5   # 点的大小

# 分辨率控制 (全局生效)
plt.rcParams["figure.dpi"] = 600         # 屏幕显示 DPI
plt.rcParams["savefig.dpi"] = 600        # 保存文件 DPI

# 颜色定义（保持主方法颜色不变；修复消融颜色冲突）
STRATEGY_COLORS = {
    # Main Baselines
    "HyperDriver": "#d62728",
    "HyperDriver (Full)": "#d62728",
    "DC": "#ff7f0e",
    "BC": "#2ca02c",
    "EC": "#9467bd",
    "Random": "#7f7f7f",
    # Ablations
    "w/o Greedy": "#8c564b",
    "w/o Hypergraph": "#bcbd22",  # [FIX] 避免与 EC(#9467bd) 冲突
    "w/o Dynamics": "#17becf",
}


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_publication_figure(fig, out_path: str) -> None:
    """
    同时保存 PDF(矢量) + PNG(预览/Word)。
    - 更稳健的后缀处理：支持 out_path 传 xxx.png / xxx.pdf / xxx（无后缀）
    - 强制白底 + 非透明，避免黑边/黑底
    """
    _ensure_parent_dir(out_path)

    base_path, ext = os.path.splitext(out_path)
    if ext.lower() in [".png", ".pdf"]:
        pass
    else:
        # out_path 没有后缀，base_path 仍然是 out_path
        base_path = out_path

    # 强制白底（对某些渲染器很关键）
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    # PDF: 矢量图，适合投稿
    fig.savefig(
        f"{base_path}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # PNG: 位图，适合 Word 插入预览
    fig.savefig(
        f"{base_path}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


# ============================
# 2. 单个数据集绘图函数
# ============================

def plot_efficiency_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "efficiency_battle.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    main_strats = ["HyperDriver (Full)", "DC", "BC", "EC", "Random"]
    df = df[df["strategy"].isin(main_strats)]
    df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"

    df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    fig = plt.figure(figsize=STD_FIG_SIZE)

    # draw_order = ["Random", "BC", "EC", "DC", "HyperDriver"]
    draw_order = ["HyperDriver", "BC", "DC", "EC", "Random"]
    strategies = df["strategy"].unique()

    for strat in draw_order:
        if strat not in strategies:
            continue
        sub = df[df["strategy"] == strat].sort_values("selected_frac")
        color = STRATEGY_COLORS.get(strat, "black")

        lw = 1.2 if strat == "HyperDriver" else 1.0
        ls = "-" if strat == "HyperDriver" else ("--" if strat == "Random" else "-.")
        marker = "o" if strat == "HyperDriver" else None
        ms = 3.5 if strat == "HyperDriver" else 0
        zorder = 10 if strat == "HyperDriver" else 1

        plt.plot(
            sub["selected_frac"],
            sub["log_energy"],
            label=strat,
            color=color,
            linewidth=lw,
            linestyle=ls,
            marker=marker,
            markersize=ms,
            zorder=zorder,
        )

    plt.xlabel("Fraction of Driver Nodes (Top 30%)")
    plt.ylabel("Log10(Control Energy Cost)")
    plt.title(f"Efficiency: {dataset_name}")
    plt.xlim(0.0, 0.3)

    plt.legend(
        frameon=True,
        loc="best",
        labelspacing=0.2,
        borderpad=0.2,
        edgecolor="gray",
        framealpha=0.9,
    ).get_frame().set_linewidth(0.8)

    plt.tight_layout()

    save_publication_figure(
        fig,
        os.path.join(root_dir, "figures", "energy_battles", f"{dataset_name}_efficiency_battle.png"),
    )
    plt.close()
    print(f"[INFO] Saved Efficiency Battle for {dataset_name}")


def plot_ablation_battle(root_dir: str, dataset_name: str):
    csv_path = os.path.join(root_dir, "results", dataset_name, "full", "efficiency_battle.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    ablation_strats = ["HyperDriver (Full)", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    df = df[df["strategy"].isin(ablation_strats)]

    # [FIX] 统一命名：与其他图一致（HyperDriver）
    df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"

    df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
    df = df[df["selected_frac"] <= 0.32]

    fig = plt.figure(figsize=STD_FIG_SIZE)

    draw_order = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    strategies = df["strategy"].unique()

    for strat in draw_order:
        if strat not in strategies:
            continue
        sub = df[df["strategy"] == strat].sort_values("selected_frac")
        color = STRATEGY_COLORS.get(strat, "black")

        if strat == "HyperDriver":
            lw, ls, marker, alpha, zorder = 1.2, "-", "o", 1.0, 10
        else:
            lw, ls, marker, alpha, zorder = 1.0, "--", None, 0.8, 1

        plt.plot(
            sub["selected_frac"],
            sub["log_energy"],
            label=strat,
            color=color,
            linewidth=lw,
            linestyle=ls,
            marker=marker,
            markersize=3.5,
            alpha=alpha,
            zorder=zorder,
        )

    plt.xlabel("Fraction of Driver Nodes (Top 30%)")
    plt.ylabel("Log10(Control Energy Cost)")
    plt.title(f"Ablation: {dataset_name}")
    plt.xlim(0.0, 0.3)

    plt.legend(
        frameon=True,
        loc="best",
        labelspacing=0.2,
        borderpad=0.2,
        edgecolor="gray",
        framealpha=0.9,
    ).get_frame().set_linewidth(0.8)

    plt.tight_layout()

    save_publication_figure(
        fig,
        os.path.join(root_dir, "figures", "ablation_battles", f"{dataset_name}_ablation_battle.png"),
    )
    plt.close()
    print(f"[INFO] Saved Ablation Battle for {dataset_name}")


def plot_top_drivers_ki(root_dir: str, dataset_name: str, top_k=10):
    """
    [Standardized] 默认展示 Top 10，统一尺寸，统一字体
    """
    scores_path = os.path.join(root_dir, "results", dataset_name, "full", "node_scores.csv")
    if not os.path.exists(scores_path):
        return
    df = pd.read_csv(scores_path)
    if "driver_score" not in df.columns:
        return

    # [FIX] nodes.csv 缺失时不崩溃：默认当作非必需（保持脚本可跑）
    if "essential" not in df.columns:
        nodes_path = os.path.join(root_dir, "processed", dataset_name, "nodes.csv")
        if os.path.exists(nodes_path):
            nodes = pd.read_csv(nodes_path)
            if "protein" in nodes.columns and "essential" in nodes.columns:
                df = df.merge(nodes[["protein", "essential"]], on="protein", how="left")
        if "essential" not in df.columns:
            df["essential"] = 0

    df = df.sort_values("driver_score", ascending=False).head(top_k).iloc[::-1]
    proteins = df["protein"].astype(str).tolist()
    vals = df["driver_score"].values
    colors = ["#7f7f7f" if e == 1 else "#ff7f0e" for e in df["essential"].fillna(0).values]

    fig = plt.figure(figsize=STD_FIG_SIZE)

    plt.barh(range(len(proteins)), vals, color=colors, height=0.65)
    plt.yticks(range(len(proteins)), proteins)

    plt.xlabel(r"Driver Score ($K_i$)")
    plt.title(f"Top {top_k} Drivers ({dataset_name})")

    legend_elements = [
        Patch(facecolor="#7f7f7f", label="Essential"),
        Patch(facecolor="#ff7f0e", label="Non-Essential"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        borderpad=0.3,
        framealpha=0.8,
        edgecolor="gray",
        fontsize=7,
    ).get_frame().set_linewidth(0.8)

    plt.tight_layout()

    save_publication_figure(
        fig,
        os.path.join(root_dir, "figures", "top_drivers", f"{dataset_name}_top_drivers.png"),
    )
    plt.close()
    print(f"[INFO] Saved Top {top_k} Drivers for {dataset_name}")


# ============================
# 3. 全局汇总绘图函数
# ============================

def plot_global_efficiency_boxplot(df: pd.DataFrame, out_dir: str):
    """全局效率箱线图"""
    if df.empty:
        return
    main_strats = ["HyperDriver", "BC", "DC", "EC", "Random"]
    sub_df = df[df["Strategy"].isin(main_strats)]

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]

    sns.boxplot(
        data=sub_df,
        x="Strategy",
        y="MeanLogEnergy",
        order=order,
        palette=STRATEGY_COLORS,
        width=0.6,
        linewidth=1.0,
        showfliers=False,
    )
    sns.stripplot(
        data=sub_df,
        x="Strategy",
        y="MeanLogEnergy",
        order=order,
        color=".3",
        size=3,
        alpha=0.6,
        jitter=True,
    )

    plt.ylabel("Mean Log10 Energy Cost")
    plt.xlabel("")
    plt.title("Global Efficiency Comparison")
    plt.xticks(rotation=20)

    plt.tight_layout()

    save_publication_figure(fig, os.path.join(out_dir, "global_efficiency_summary.png"))
    plt.close()
    print("[INFO] Saved Global Efficiency Summary")


def plot_global_ablation_boxplot(df: pd.DataFrame, out_dir: str):
    """全局消融箱线图"""
    if df.empty:
        return
    ablation_strats = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]
    sub_df = df[df["Strategy"].isin(ablation_strats)]

    fig = plt.figure(figsize=STD_FIG_SIZE)
    order = ["HyperDriver", "w/o Greedy", "w/o Hypergraph", "w/o Dynamics"]

    sns.boxplot(
        data=sub_df,
        x="Strategy",
        y="MeanLogEnergy",
        order=order,
        palette=STRATEGY_COLORS,
        width=0.6,
        linewidth=1.0,
        showfliers=False,
    )
    sns.stripplot(
        data=sub_df,
        x="Strategy",
        y="MeanLogEnergy",
        order=order,
        color=".3",
        size=3,
        alpha=0.6,
        jitter=True,
    )

    plt.ylabel("Mean Log10 Energy Cost")
    plt.xlabel("")
    plt.title("Global Ablation Summary")
    plt.xticks(rotation=20)

    plt.tight_layout()

    save_publication_figure(fig, os.path.join(out_dir, "global_ablation_summary.png"))
    plt.close()
    print("[INFO] Saved Global Ablation Summary")


def plot_global_driver_composition(df: pd.DataFrame, out_dir: str):
    """全局成分箱线图"""
    if df.empty:
        return

    fig = plt.figure(figsize=STD_FIG_SIZE)

    # [FIX] 顺序与全局效率图一致，提升可读性
    order = ["HyperDriver", "BC", "DC", "EC", "Random"]

    sns.boxplot(
        data=df,
        x="Strategy",
        y="EssentialFrac",
        order=order,
        palette=STRATEGY_COLORS,
        width=0.6,
        linewidth=1.0,
        showfliers=False,
    )
    sns.stripplot(
        data=df,
        x="Strategy",
        y="EssentialFrac",
        order=order,
        color=".3",
        size=3,
        alpha=0.6,
        jitter=True,
    )

    plt.ylabel("Essential Fraction (in Top-20)")
    plt.xlabel("")
    plt.title("Driver Essentiality Composition")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)

    plt.tight_layout()

    save_publication_figure(fig, os.path.join(out_dir, "global_driver_composition.png"))
    plt.close()
    print("[INFO] Saved Global Composition Summary")


# ============================
# 4. 数据收集逻辑
# ============================

def collect_efficiency_metrics(root_dir: str, dataset_list: list) -> pd.DataFrame:
    records = []
    for ds in dataset_list:
        csv_path = os.path.join(root_dir, "results", ds, "full", "efficiency_battle.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df.loc[df["strategy"] == "HyperDriver (Full)", "strategy"] = "HyperDriver"
        df = df[df["selected_frac"] <= 0.32]
        df["log_energy"] = np.log10(df["energy_cost"] + 1e-9)
        grouped = df.groupby("strategy")["log_energy"].mean()
        for strat, score in grouped.items():
            records.append({"Dataset": ds, "Strategy": strat, "MeanLogEnergy": score})
    return pd.DataFrame(records)


def collect_driver_composition(root_dir: str, dataset_list: list, top_k: int = 20) -> pd.DataFrame:
    records = []
    for ds in dataset_list:
        nodes_path = os.path.join(root_dir, "processed", ds, "nodes.csv")
        if not os.path.exists(nodes_path):
            continue
        nodes_df = pd.read_csv(nodes_path)
        if "essential" not in nodes_df.columns:
            continue
        ess_labels = nodes_df["essential"].values

        # HyperDriver
        hd_path = os.path.join(root_dir, "results", ds, "full", "node_scores.csv")
        if os.path.exists(hd_path):
            hd_df = pd.read_csv(hd_path)
            if "driver_score" in hd_df.columns:
                top_idx = np.argsort(-hd_df["driver_score"].values)[:top_k]
                ess_frac = ess_labels[top_idx].mean()
                records.append({"Dataset": ds, "Strategy": "HyperDriver", "EssentialFrac": ess_frac})

        # Baselines
        base_dir = os.path.join(root_dir, "results", ds, "baselines")
        for method in ["dc", "bc", "ec"]:
            path = os.path.join(base_dir, f"{method}_scores.csv")
            if os.path.exists(path):
                b_df = pd.read_csv(path)
                if method in b_df.columns:
                    top_idx = np.argsort(-b_df[method].values)[:top_k]
                    ess_frac = ess_labels[top_idx].mean()
                    records.append({"Dataset": ds, "Strategy": method.upper(), "EssentialFrac": ess_frac})

        # Random
        global_ratio = ess_labels.mean()
        records.append({"Dataset": ds, "Strategy": "Random", "EssentialFrac": global_ratio})

    return pd.DataFrame(records)


def plot_global_summaries(root_dir: str, dataset_list: list):
    """Wrapper for all global plots"""
    print("  -> Collecting Global Data...")
    eff_df = collect_efficiency_metrics(root_dir, dataset_list)
    comp_df = collect_driver_composition(root_dir, dataset_list)
    out_dir = os.path.join(root_dir, "figures", "global_summary")
    os.makedirs(out_dir, exist_ok=True)

    if not eff_df.empty:
        plot_global_efficiency_boxplot(eff_df, out_dir)
        plot_global_ablation_boxplot(eff_df, out_dir)
    else:
        print("[WARN] No efficiency data found.")

    if not comp_df.empty:
        plot_global_driver_composition(comp_df, out_dir)
    else:
        print("[WARN] No composition data found.")


# ============================
# 主入口
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--fig", type=str, default="all")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset == "all":
        dataset_list = load_datasets_config(os.path.join(root_dir, "conf", "datasets.json"))
    else:
        dataset_list = [args.dataset]

    print("========== Generating Nature Figures (3.5x3.0 / Unified Box) ==========")
    for ds in dataset_list:
        plot_efficiency_battle(root_dir, ds)
        plot_ablation_battle(root_dir, ds)
        plot_top_drivers_ki(root_dir, ds, top_k=10)

    if args.dataset == "all":
        plot_global_summaries(root_dir, dataset_list)


if __name__ == "__main__":
    main()
