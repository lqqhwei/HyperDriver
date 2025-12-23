import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 导入之前的模块
from data_loader import DataLoader
from utils_graph import SubgraphSelector
from methods import DriverSelector
from control_energy import ControlEnergyCalculator

# 配置
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "resource/energy_case_study/output"
STATIC_DIR = ROOT / "data/Yu"
LABEL_DIR = ROOT / "data"
TOP_K = 10               # 在子图中选 10 个驱动节点
RANDOM_RUNS = 20         # 随机策略跑20次取平均

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_results(results, subgraph_name="Case_Study"):
    """
    绘制能量对比柱状图 (Log Scale)
    并保存为 PNG 和 PDF (矢量图)
    """
    methods = list(results.keys())
    energies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    
    # 颜色配置: HyperDriver 用醒目的红色，其他用蓝色/灰色
    colors = ['#d62728' if 'HyperDriver' in m else '#1f77b4' for m in methods]
    
    bars = plt.bar(methods, energies, color=colors, alpha=0.8, edgecolor='black')
    
    plt.ylabel('Required Control Energy ($J_{avg}$)', fontsize=12)
    plt.title(f'True Minimum Control Energy Comparison\n(Subgraph Size: {subgraph_name})', fontsize=14)
    plt.yscale('log') # 关键：对数坐标
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.1, 
                 f'{yval:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # --- 保存 PNG (位图) ---
    save_path_png = os.path.join(OUTPUT_DIR, 'energy_comparison_case_study.png')
    plt.savefig(save_path_png, dpi=300)
    print(f"📊 Plot saved to {save_path_png}")

    # --- 保存 PDF (矢量图, 论文专用) ---
    save_path_pdf = os.path.join(OUTPUT_DIR, 'energy_comparison_case_study.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    print(f"📄 Vector plot saved to {save_path_pdf}")
    
    plt.close()

def main():
    setup_dirs()
    print("🚀 Starting True Minimum Energy Case Study Experiment...")
    
    # 1. 加载全图
    loader = DataLoader(STATIC_DIR,LABEL_DIR)
    G_full = loader.load_static_graph()
    
    # 2. 提取 Case Study 子图
    # 寻找一个 60-120 个节点的致密功能模块
    selector = SubgraphSelector(G_full, target_size=(60, 120))
    G_sub, A_sub, sub_nodes, sub_map = selector.get_best_subgraph()
    
    print(f"\n🔬 Case Study Subgraph Selected:")
    print(f"   - Nodes: {len(sub_nodes)}")
    print(f"   - Edges: {G_sub.number_of_edges()}")
    
    # 3. 初始化选择器和计算器
    driver_selector = DriverSelector(G_sub, A_sub, sub_nodes)
    energy_calc = ControlEnergyCalculator(A_sub)
    
    results = {}
    
    # ==========================
    # Method 1: HyperDriver (Ours)
    # ==========================
    print(f"\n[1] Running HyperDriver (Spectral Proxy)...")
    idx_hd, names_hd = driver_selector.select_hyperdriver_proxy(TOP_K)
    energy_hd = energy_calc.compute_energy(idx_hd)
    results['HyperDriver'] = energy_hd
    print(f"   - Selected: {names_hd[:3]}...")
    print(f"   - Energy: {energy_hd:.4e}")

    # ==========================
    # Method 2: Degree Centrality (Hubs)
    # ==========================
    print(f"\n[2] Running Degree Centrality (Hubs)...")
    idx_deg, names_deg = driver_selector.select_degree(TOP_K)
    energy_deg = energy_calc.compute_energy(idx_deg)
    results['Degree (Hub)'] = energy_deg
    print(f"   - Selected: {names_deg[:3]}...")
    print(f"   - Energy: {energy_deg:.4e}")

    # ==========================
    # Method 3: Random Baseline (Avg)
    # ==========================
    print(f"\n[3] Running Random Selection ({RANDOM_RUNS} runs)...")
    random_energies = []
    for _ in tqdm(range(RANDOM_RUNS)):
        idx_rnd, _ = driver_selector.select_random(TOP_K)
        e = energy_calc.compute_energy(idx_rnd)
        if e != float('inf'):
            random_energies.append(e)
    
    avg_random_energy = np.mean(random_energies) if random_energies else float('inf')
    results['Random'] = avg_random_energy
    print(f"   - Avg Energy: {avg_random_energy:.4e}")

    # ==========================
    # 4. 结果汇总与导出 (CSV + PNG + PDF)
    # ==========================
    print("\n🏆 Final Results Summary (Lower Energy is Better):")
    
    # 创建 DataFrame
    df_res = pd.DataFrame(list(results.items()), columns=['Method', 'Control_Energy_J'])
    
    # 打印到控制台
    print(df_res)
    
    # 保存为 CSV 文件
    csv_filename = f"energy_results_{len(sub_nodes)}_nodes.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    df_res.to_csv(csv_path, index=False)
    print(f"💾 Data saved to CSV: {csv_path}")

    # 绘制图片 (PNG & PDF)
    plot_results(results, subgraph_name=f"{len(sub_nodes)}_nodes")

if __name__ == "__main__":
    main()