"""
Qiskit 版 VQE — H2 分子基態能量
===============================

用 Qiskit 的 StatevectorEstimator 計算 ansatz 電路的能量期望值，
再用 scipy COBYLA 調整參數，逼近 H2 的基態能量。

執行方式：
    python quantum_vqe/run_qiskit.py

輸出：
    - 終端機顯示收斂過程與結果
    - outputs/qiskit_convergence.csv（每代能量）
    - outputs/qiskit_energy.png（收斂曲線）
"""

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from h2_hamiltonian import INITIAL_PARAMS, PAULI_TERMS, exact_ground_state_energy

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_ansatz() -> QuantumCircuit:
    """4 參數的 hardware-efficient ansatz。"""
    theta = ParameterVector("θ", 4)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    return qc


def build_observable() -> SparsePauliOp:
    return SparsePauliOp.from_list([(p, c) for p, c in PAULI_TERMS])


def main():
    print("=" * 60)
    print("Qiskit VQE — H2 基態能量")
    print("=" * 60)

    ansatz = build_ansatz()
    observable = build_observable()
    estimator = StatevectorEstimator()

    exact = exact_ground_state_energy()
    print(f"精確基態能量（參考值）: {exact:.6f} Ha")

    def energy(params):
        result = estimator.run([(ansatz, observable, [params])]).result()
        return float(result[0].data.evs[0])

    # 紀錄每一代的能量（COBYLA 的 callback 每代呼叫一次）
    history: list[float] = []

    def callback(xk):
        history.append(energy(xk))

    print(f"初始參數: {INITIAL_PARAMS}")
    print(f"初始能量: {energy(INITIAL_PARAMS):.6f} Ha")
    print("開始最佳化（COBYLA）...")

    res = minimize(
        energy,
        INITIAL_PARAMS,
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
    np.savetxt(OUT_DIR / "qiskit_convergence.csv", np.array(history), header="energy", comments="")

    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(history, color="#1f77b4", label="Qiskit VQE")
    plt.axhline(exact, color="green", ls="--", label=f"精確值 {exact:.4f}")
    plt.xlabel("最佳化代數 (iteration)")
    plt.ylabel("能量 (Ha)")
    plt.title("Qiskit VQE — H2 能量收斂")
    plt.legend()
    plt.tight_layout()
    png = OUT_DIR / "qiskit_energy.png"
    plt.savefig(png, dpi=120)
    plt.close()
    print(f"\n[圖表] 已儲存: {png}")


if __name__ == "__main__":
    main()
