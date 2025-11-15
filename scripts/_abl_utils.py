# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GDIR = ROOT/"outputs"/"graphs"
HDIR = ROOT/"outputs"/"hypergraphs"
RES  = ROOT/"outputs"/"results"
FIGS = RES /"figs"

def _load_series(prefix: str):
    files = sorted(GDIR.glob(f"{prefix}_t_*.npy"))
    if not files: files = sorted(HDIR.glob(f"{prefix}_t_*.npy"))
    return [np.load(f) for f in files]

def load_L1_L2_Z():
    L1 = _load_series("L1")
    L2 = _load_series("L2")
    Zs = _load_series("embeds_Z")
    T = min(len(L1), len(L2), len(Zs))
    L1, L2, Zs = L1[:T], L2[:T], Zs[:T]
    N = Zs[0].shape[0]
    return T, N, L1, L2, Zs

def compute_S_from_Z(Zs, clip_q=0.995):
    S = []
    for Z in Zs:
        s = (Z*Z).sum(axis=1)  # ||Z_i||^2
        cap = np.quantile(s, clip_q)
        s = np.clip(s, 0.0, cap)
        S.append(s.astype(np.float32))
    return S  # list[T] of (N,)

def compute_E_via_pinv(L1, L2, a1, a2, delta, use_cg=False, cg_it=200, cg_tol=1e-6):
    """返回 list[T] of diag((-A)^-1), 其中 A=-(a1 L1 + a2 L2) - delta I."""
    T = len(L1); E = []
    for t in range(T):
        L = a1*L1[t] + a2*L2[t]
        A = -L - delta*np.eye(L.shape[0], dtype=np.float64)
        if not use_cg:
            # pinv：Hazbun 足够快
            Minv = np.linalg.pinv(-A, rcond=1e-6)
            e = np.diag(Minv).astype(np.float32)
        else:
            # CG Hutchinson 估计对角（更省内存），保留接口以便将来放大数据集
            import numpy.linalg as npl
            from numpy.random import default_rng
            rng = default_rng(42)
            nprobe = 8
            e = np.zeros(A.shape[0], dtype=np.float64)
            # 预分解（简易对角预条件）
            Mdiag = np.diag(-A); Mdiag = np.where(np.abs(Mdiag)>1e-12, Mdiag, 1.0)
            MinvD = 1.0/Mdiag
            for _ in range(nprobe):
                z = rng.choice([-1.0, 1.0], size=A.shape[0])
                # 共轭梯度
                x = np.zeros_like(z); r = z - (-A)@x; p = MinvD*r; rs = r@p
                for _it in range(cg_it):
                    Ap = (-A)@p
                    alpha = rs/(p@Ap + 1e-12)
                    x = x + alpha*p
                    r = r - alpha*Ap
                    if npl.norm(r) < cg_tol: break
                    s = MinvD*r
                    rs_new = r@s
                    p = s + (rs_new/rs)*p
                    rs = rs_new
                e += x*z
            e = (e/nprobe).astype(np.float32)
        E.append(e)
    return E

def K_from_S_E(S_list, E_list, eps=1e-6):
    K = []
    for S, E in zip(S_list, E_list):
        K.append((S/(E+eps)).astype(np.float32))
    return np.stack(K, axis=0)  # (T,N)

def save_rank_and_plots(K, nodes, tag: str, top_frac=0.1):
    T, N = K.shape
    # 全局聚合
    k_mean = K.mean(axis=0)
    k_max  = K.max(axis=0)
    df = pd.DataFrame({"Node": nodes, "K_mean": k_mean, "K_max": k_max})
    df["rank_by_Kmean"] = df["K_mean"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values("rank_by_Kmean")
    out_csv = RES/f"{tag}_rank_global.csv"
    df.to_csv(out_csv, index=False)

    # TopN 柱图
    topN = df.nsmallest(30, "rank_by_Kmean")
    plt.figure(figsize=(14,5))
    plt.bar(topN["Node"], topN["K_mean"])
    plt.xticks(rotation=65, ha="right"); plt.ylabel("k_mean"); plt.title("Top-30 by k_mean")
    plt.tight_layout(); plt.savefig(FIGS/f"{tag}_rank_topN.png", dpi=180); plt.close()

    # 时间曲线（同你现在 Step 5 的改法）
    mean_K   = K.mean(axis=1)
    median_K = np.median(K, axis=1)
    topK = max(1, int(top_frac*N))
    idx_topK = np.argpartition(K, -topK, axis=1)[:, -topK:]
    topK_mean_K = np.take_along_axis(K, idx_topK, axis=1).mean(axis=1)
    t = np.arange(1, T+1)
    plt.figure(figsize=(10,5))
    plt.plot(t, mean_K, label="mean_K")
    plt.plot(t, median_K, label="median_K")
    plt.plot(t, topK_mean_K, label="topK_mean_K")
    plt.xlabel("Time t"); plt.ylabel("K score")
    plt.title(f"{tag}: K-score summary over time")
    plt.legend(); plt.tight_layout(); plt.savefig(FIGS/f"{tag}_kstats.png", dpi=180); plt.close()

    # 稳定性
    def overlap_at_k(a,b,k):
        A = set(np.argpartition(a,-k)[-k:]); B = set(np.argpartition(b,-k)[-k:])
        return len(A&B)/float(k)
    overlap = [np.nan] + [overlap_at_k(K[t-1], K[t], topK) for t in range(1,T)]
    def spearman_like(a,b):
        ra = np.argsort(np.argsort(-a)); rb = np.argsort(np.argsort(-b))
        return np.corrcoef(ra,rb)[0,1]
    rho = [np.nan] + [spearman_like(K[t-1], K[t]) for t in range(1,T)]
    plt.figure(figsize=(10,5))
    plt.plot(t, overlap, label="TopK overlap w/ prev")
    plt.plot(t, rho, label="Spearman w/ prev")
    plt.ylim(-0.1, 1.05); plt.xlabel("Time t"); plt.ylabel("stability")
    plt.title(f"{tag}: ranking stability"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS/f"{tag}_stability.png", dpi=180); plt.close()

    print(f"[OK] {tag} 输出：{out_csv}，以及 {FIGS} 下的三张图")
    return out_csv


# ====== RES UTILS: common helpers for RESations (safe to append) ======
# 本段为向后兼容扩展：若已有同名符号则不覆盖，保证老脚本仍可用。

# ---- 数值稳健工具：对称化 + float64 ----
try:
    sym64  # noqa: F821
except NameError:
    def sym64(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.float64, order="C")
        return 0.5 * (M + M.T)

try:
    symmetrize_series  # noqa: F821
except NameError:
    def symmetrize_series(M_list):
        return [sym64(M) for M in M_list]

# ---- 混合/自动回退的 E(t) 计算包装器 ----
# 依赖于上面的 compute_E_via_pinv(L1s, L2s, a1, a2, delta, use_cg)
try:
    compute_E_hybrid  # noqa: F821
except NameError:
    def compute_E_hybrid(
        L1_list, L2_list, *,
        a1: float = 0.5, a2: float = 0.5,
        delta: float = 5e-3,
        small_n: int = 800,
        force_cg: bool = False
    ):
        """
        小图 (n<=small_n) 优先 SVD 伪逆；大图或 SVD 失败自动回退到 CG/Hutchinson。
        返回: E(t) 列表/数组（与 compute_E_via_pinv 的返回一致）
        """
        L1s = symmetrize_series(L1_list)
        L2s = symmetrize_series(L2_list)
        n = L1s[0].shape[0]

        if (not force_cg) and (n <= small_n):
            try:
                print(f"[INFO] try SVD pinv (n={n}, a1={a1:.2f}, a2={a2:.2f}) ...")
                return compute_E_via_pinv(L1s, L2s, a1=a1, a2=a2, delta=delta, use_cg=False)
            except Exception as e:
                print(f"[WARN] SVD path failed: {type(e).__name__}: {e} -> fallback CG")

        print(f"[INFO] use CG/Hutchinson path (n={n}, a1={a1:.2f}, a2={a2:.2f}) ...")
        return compute_E_via_pinv(L1s, L2s, a1=a1, a2=a2, delta=delta, use_cg=True)
