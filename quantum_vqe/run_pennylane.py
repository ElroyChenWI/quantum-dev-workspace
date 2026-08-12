"""
PennyLane 版 VQE — H2 分子基態能量
==================================

用 PennyLane 的 default.qubit 裝置計算 ansatz 的期望值，
再用 scipy COBYLA 調整參數。與 Qiskit 版共用同一個哈密頓量定義。

執行方式：
    python quantum_vqe/run_pennylane.py

輸出：
    - 終端機顯示收斂過程與結果
    - outputs/pennylane_convergence.csv（每代能量）
    - outputs/pennylane_energy.png（收斂曲線）
"""

from pathlib import Path

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from h2_hamiltonian import INITIAL_PARAMS, PAULI_TERMS, exact_ground_state_energy

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_hamiltonian():
    """從共用 PAULI_TERMS 建構 PennyLane 哈密頓量。

    pauli 字串 e.g. "ZI" → Z 在 wire 0、I 在 wire 1（與 numpy/Qiskit 一致）。
    """
    pauli_gate = {
        "I": qml.Identity,
        "X": qml.PauliX,
        "Y": qml.PauliY,
        "Z": qml.PauliZ,
    }
    ops, coeffs = [], []
    for pauli, coef in PAULI_TERMS:
        op = None
        for wire, ch in enumerate(pauli):
            gate = pauli_gate[ch](wire)
            op = gate if op is None else op @ gate
        ops.append(op)
        coeffs.append(coef)
    return qml.dot(coeffs, ops)


def main():
    print("=" * 60)
    print("PennyLane VQE — H2 基態能量")
    print("=" * 60)

    dev = qml.device("default.qubit", wires=2)
    H = build_hamiltonian()
    exact = exact_ground_state_energy()
    print(f"精確基態能量（參考值）: {exact:.6f} Ha")

    @qml.qnode(dev)
    def circuit(params):
        # 與 Qiskit 版完全相同的 ansatz
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        return qml.expval(H)

    def energy(params):
        return float(circuit(np.array(params, dtype=float)))

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
    np.savetxt(OUT_DIR / "pennylane_convergence.csv", np.array(history), header="energy", comments="")

    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(history, color="#ff7f0e", label="PennyLane VQE")
    plt.axhline(exact, color="green", ls="--", label=f"精確值 {exact:.4f}")
    plt.xlabel("最佳化代數 (iteration)")
    plt.ylabel("能量 (Ha)")
    plt.title("PennyLane VQE — H2 能量收斂")
    plt.legend()
    plt.tight_layout()
    png = OUT_DIR / "pennylane_energy.png"
    plt.savefig(png, dpi=120)
    plt.close()
    print(f"\n[圖表] 已儲存: {png}")


if __name__ == "__main__":
    main()
