"""
CUDA-Q 版 VQE — H2 分子基態能量
===============================

重要：CUDA-Q 不支援 Windows 原生執行！
請在 WSL2 / Linux / Docker 環境中執行本腳本，安裝步驟見 docs/CUDA-Q_SETUP.md。

這個版本與 Qiskit / PennyLane 版共用同一個哈密頓量與 ansatz，
只是把「量子執行引擎」換成 CUDA-Q（可吃 NVIDIA GPU 加速）。

執行方式（在 WSL2 / Docker 內）：
    python quantum_vqe/run_cudaq.py

輸出：
    - 終端機顯示收斂過程與結果
    - outputs/cudaq_convergence.csv（每代能量）
    - outputs/cudaq_energy.png（收斂曲線）
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    import cudaq
    from cudaq import spin
except ImportError:
    print("CUDA-Q 未安裝。此腳本需在 WSL2 / Linux / Docker 環境執行。")
    print("安裝步驟請參考專案根目錄的 docs/CUDA-Q_SETUP.md")
    sys.exit(1)

# 讓腳本無論從哪個目錄執行都能找到共用模組
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2_hamiltonian import INITIAL_PARAMS, PAULI_TERMS, exact_ground_state_energy

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_hamiltonian():
    """把共用 PAULI_TERMS 轉成 CUDA-Q 的 spin operator。

    pauli 字串 e.g. "ZI" → spin.z(0) * spin.i(1)。
    """
    H = None
    for pauli, coef in PAULI_TERMS:
        term = None
        for wire, ch in enumerate(pauli):
            if ch == "I":
                single = spin.i(wire)
            elif ch == "X":
                single = spin.x(wire)
            elif ch == "Y":
                single = spin.y(wire)
            elif ch == "Z":
                single = spin.z(wire)
            else:
                raise ValueError(f"未知的 Pauli 字元: {ch}")
            term = single if term is None else term * single
        H = coef * term if H is None else H + coef * term
    return H


@cudaq.kernel
def ansatz(params: list[float]):
    """與 Qiskit / PennyLane 版完全相同的 ansatz。"""
    q = cudaq.qvector(2)
    ry(params[0], q[0])
    ry(params[1], q[1])
    x.ctrl(q[0], q[1])   # CNOT
    ry(params[2], q[0])
    ry(params[3], q[1])


def main():
    print("=" * 60)
    print("CUDA-Q VQE — H2 基態能量")
    print("=" * 60)
    print(f"CUDA-Q 版本: {cudaq.__version__}")
    print(f"執行目標   : {cudaq.get_target()}")

    hamiltonian = build_hamiltonian()
    exact = exact_ground_state_energy()
    print(f"精確基態能量（參考值）: {exact:.6f} Ha")

    def energy(params):
        result = cudaq.observe(ansatz, hamiltonian, list(params))
        return float(result.expectation())

    history: list[float] = []

    def callback(xk):
        history.append(energy(xk))

    print(f"初始參數: {INITIAL_PARAMS}")
    print(f"初始能量: {energy(INITIAL_PARAMS):.6f} Ha")
    print("開始最佳化（COBYLA）...")

    res = minimize(
        energy,
        np.array(INITIAL_PARAMS, dtype=float),
        method="COBYLA",
        callback=callback,
        options={"maxiter": 300, "tol": 1e-6},
    )

    final_e = float(res.fun)
    print(f"\n最佳化完成，用了 {len(history)} 代")
    print(f"VQE 得到基態能量: {final_e:.6f} Ha")
    print(f"精確值           : {exact:.6f} Ha")
    print(f"誤差             : {abs(final_e - exact):.2e} Ha")

    # 儲存收斂資料 + 畫圖
    OUT_DIR.mkdir(exist_ok=True)
    np.savetxt(OUT_DIR / "cudaq_convergence.csv", np.array(history), header="energy", comments="")

    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(history, color="#2ca02c", label="CUDA-Q VQE")
    plt.axhline(exact, color="green", ls="--", label=f"精確值 {exact:.4f}")
    plt.xlabel("最佳化代數 (iteration)")
    plt.ylabel("能量 (Ha)")
    plt.title("CUDA-Q VQE — H2 能量收斂")
    plt.legend()
    plt.tight_layout()
    png = OUT_DIR / "cudaq_energy.png"
    plt.savefig(png, dpi=120)
    plt.close()
    print(f"\n[圖表] 已儲存: {png}")


if __name__ == "__main__":
    main()
