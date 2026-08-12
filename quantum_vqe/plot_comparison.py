"""
VQE 三框架收斂比較圖
====================

讀取三個框架的收斂資料（outputs/*_convergence.csv），畫在同一張圖比較。
已跑過的框架就畫出來；沒跑過的（例如 CUDA-Q 還沒裝）會自動略過。

執行方式：
    python quantum_vqe/plot_comparison.py
"""

from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2_hamiltonian import exact_ground_state_energy

OUT_DIR = Path(__file__).resolve().parent / "outputs"

FRAMEWORKS = [
    ("qiskit", "Qiskit", "#1f77b4"),
    ("pennylane", "PennyLane", "#ff7f0e"),
    ("cudaq", "CUDA-Q", "#2ca02c"),
]


def load(name):
    csv_path = OUT_DIR / f"{name}_convergence.csv"
    if not csv_path.exists():
        return None
    # CSV 第一行是 header "energy"，跳過
    data = np.loadtxt(csv_path, skiprows=1)
    return np.atleast_1d(data)  # 只有一列時 np.loadtxt 回傳 scalar，統一成 array


def main():
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    exact = exact_ground_state_energy()

    plt.figure(figsize=(9, 5.5))
    plotted = False
    for name, label, color in FRAMEWORKS:
        data = load(name)
        if data is None:
            print(f"[略過] {label}：尚未執行（outputs/{name}_convergence.csv 不存在）")
            continue
        plt.plot(data, color=color, label=f"{label} VQE")
        plotted = True
        print(f"[讀取] {label}：{len(data)} 代，最終能量 = {data[-1]:.6f} Ha")

    if not plotted:
        print("沒有任何收斂資料。請先執行 run_qiskit.py / run_pennylane.py / run_cudaq.py。")
        return

    plt.axhline(exact, color="red", ls="--", lw=1.5, label=f"精確基態能量 {exact:.4f} Ha")
    plt.xlabel("最佳化代數 (iteration)")
    plt.ylabel("能量 (Ha)")
    plt.title("VQE — H2 基態能量：三框架收斂比較")
    plt.legend()
    plt.tight_layout()

    png = OUT_DIR / "vqe_comparison.png"
    plt.savefig(png, dpi=130)
    plt.close()
    print(f"\n[圖表] 已儲存: {png}")


if __name__ == "__main__":
    main()
