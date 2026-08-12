"""
VQE 規模實驗：一路加大，直到撞到「指數牆」
============================================

從 H2 分子（2 qubit）出發，改用可任意擴展的 1D Heisenberg 鏈哈密頓量，
把 qubit 數一路加大，量測「單次能量評估的時間」與「狀態向量記憶體」，
親眼看到經典模擬的指數成長，並外推本機的「牆」在哪裡。

   H = J * Σ_{i} ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} )

   N qubit → 狀態向量大小 = 2^N 個複數 = 2^N × 16 bytes

執行方式：
    python quantum_vqe/scale_experiment.py
    python quantum_vqe/scale_experiment.py --max_qubits 24   # 預設上限
    python quantum_vqe/scale_experiment.py --max_qubits 28   # 更接近牆（小心記憶體）

輸出：
    - 終端機：時間/記憶體對照表 + 牆的預測
    - outputs/scaling_plot.png：指數成長圖
    - outputs/scaling_results.csv：原始數據
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# 每個複數（complex128）= 16 bytes
BYTES_PER_AMPLITUDE = 16


def build_heisenberg(n: int, j: float = 1.0):
    """1D Heisenberg 鏈，N qubit → 3(N-1) 個 Pauli 項。"""
    from qiskit.quantum_info import SparsePauliOp

    terms: dict[str, float] = {}
    for i in range(n - 1):
        for p0, p1 in (("X", "X"), ("Y", "Y"), ("Z", "Z")):
            label = ["I"] * n
            label[i] = p0
            label[i + 1] = p1
            key = "".join(label)
            terms[key] = terms.get(key, 0.0) + j
    return SparsePauliOp.from_list(list(terms.items()))


def build_ansatz(n: int, layers: int = 2):
    """Hardware-efficient ansatz：Ry 層 + CNOT 鏈。"""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("t", n * (layers + 1))
    qc = QuantumCircuit(n)
    idx = 0
    for i in range(n):
        qc.ry(theta[idx], i)
        idx += 1
    for _ in range(layers):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(theta[idx], i)
            idx += 1
    return qc


def main():
    parser = argparse.ArgumentParser(description="VQE 規模實驗：找到指數牆")
    parser.add_argument("--max_qubits", type=int, default=24,
                        help="最大 qubit 數（預設 24，28 以上小心記憶體）")
    parser.add_argument("--layers", type=int, default=2, help="ansatz 層數")
    parser.add_argument("--vqe_small", type=int, default=8,
                        help="對小系統跑真 VQE 的上限（預設 8）")
    args = parser.parse_args()

    from qiskit.primitives import StatevectorEstimator

    est = StatevectorEstimator()
    print("=" * 66)
    print("VQE 規模實驗 — 找指數牆")
    print("=" * 66)
    print("系統記憶體限制下的狀態向量大小（理論）：")
    print(f"  2^{args.max_qubits} qubits → 2^{args.max_qubits} × {BYTES_PER_AMPLITUDE} B "
          f"= {2**args.max_qubits * BYTES_PER_AMPLITUDE / 2**30:.2f} GB")
    print()

    results = []

    # ---- 1) 小系統：跑真正的 VQE，確認收斂 ----
    if args.vqe_small >= 2:
        from scipy.optimize import minimize

        print("--- 小系統 VQE（確認演算法仍收斂）---")
        for n in [2, 4, 6]:
            if n > args.vqe_small:
                break
            obs = build_heisenberg(n)
            qc = build_ansatz(n, args.layers)
            n_params = qc.num_parameters
            rng = np.random.default_rng(n)
            x0 = rng.uniform(-0.1, 0.1, n_params)

            def energy(p):
                return float(est.run([(qc, obs, [p])]).result()[0].data.evs[0])

            res = minimize(energy, x0, method="COBYLA",
                           options={"maxiter": 200, "tol": 1e-4})
            print(f"  N={n:2d} qubits  ({qc.num_qubits}): "
                  f"VQE 能量 = {res.fun:+.4f} (初始 {energy(x0):+.4f})")
        print()

    # ---- 2) 量測單次能量評估的時間 ----
    print("--- 單次能量評估：時間 vs qubit 數 ---")
    ns = list(range(2, args.max_qubits + 1, 2))
    for n in ns:
        obs = build_heisenberg(n)
        qc = build_ansatz(n, args.layers)
        params = np.zeros(qc.num_parameters)

        # warm up + 量測（取 3 次中位數，避免抖動）
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            est.run([(qc, obs, [params])]).result()
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        mem_gb = 2**n * BYTES_PER_AMPLITUDE / 2**30
        results.append((n, t, mem_gb))
        print(f"  N={n:2d} qubits | 時間 {t:8.4f} s | 狀態向量 {mem_gb:8.2f} GB")

    # ---- 3) 外推牆的位置 ----
    print("\n--- 指數牆外推 ---")
    n, t, mem = zip(*results)
    # 用最後幾點 fit log(t) ~ a*N + b
    k = min(5, len(results))
    a, b = np.polyfit(n[-k:], np.log(t[-k:]), 1)
    n_wall_time = (np.log(60) - b) / a        # 單次評估要 1 分鐘的 N
    n_wall_16gb = np.log(16 / 0.25) / np.log(2) + 24  # 記憶體 16 GB 的 N（粗估）
    print(f"  單次能量評估每加 2 qubit，時間約乘 {np.exp(2*a):.1f} 倍")
    print(f"  依此趨勢：N≈{n_wall_time:.0f} 時，單次評估就要 1 分鐘")
    print(f"  狀態向量超過 16 GB 約在 N≈{n_wall_16gb:.0f}")
    print(f"  這就是本機經典模擬的『指數牆』——再上去只能靠雲端/真硬體")
    print(f"  （本機實際能跑到的上限，取決於你的 RAM：")
    print(f"    看到『記憶體不足』或系統變慢的那一刻，就是你的牆）")

    # ---- 4) 儲存 + 畫圖 ----
    OUT_DIR.mkdir(exist_ok=True)
    np.savetxt(OUT_DIR / "scaling_results.csv",
               np.array(results), header="qubits,time_s,mem_gb", comments="")

    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # 用 ASCII 指數格式（如 3.0e-03），避免 log 軸上標負號的字形問題
    def ascii_sci(v, pos):
        return f"{v:.1e}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n, t, mem = zip(*results)

    axes[0].plot(n, t, "o-", color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(ascii_sci))
    axes[0].set_xlabel("Qubit 數 N")
    axes[0].set_ylabel("單次能量評估時間 (s, log)")
    axes[0].set_title("時間指數成長（log 尺規 = 直線）")

    axes[1].plot(n, mem, "o-", color="#d62728")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(ascii_sci))
    axes[1].axhline(16, color="gray", ls="--", label="16 GB（桌機常見上限）")
    axes[1].set_xlabel("Qubit 數 N")
    axes[1].set_ylabel("狀態向量記憶體 (GB, log)")
    axes[1].set_title("記憶體指數成長（指數牆）")
    axes[1].legend()

    plt.tight_layout()
    png = OUT_DIR / "scaling_plot.png"
    plt.savefig(png, dpi=130)
    plt.close()
    print(f"\n[圖表] 已儲存: {png}")


if __name__ == "__main__":
    main()
