"""
02 量子機器學習 (QML) — PennyLane 與 Qiskit
==========================================

內容：
1. PennyLane：可微分的量子程式，訓練單一 qubit 旋轉去逼近目標態
2. Qiskit VQC：量子變分分類器，做二元分類

執行方式：
    python examples/02_quantum_machine_learning.py
"""

import numpy as np


def demo_pennylane_gradient():
    """PennyLane：使用梯度下降讓量子電路的輸出逼近目標。"""
    import pennylane as qml
    from pennylane import numpy as pnp  # PennyLane 的 numpy（支援 autograd）

    print("=" * 60)
    print("PennyLane 版本：", end="")
    import pennylane
    print(pennylane.__version__)
    print("1) 訓練一個量子電路讓 |ψ(θ)⟩ 逼近 |1⟩")

    dev = qml.device("default.qubit", wires=1)
    target = np.array([0.0, 1.0])  # 目標是 |1⟩ 的機率分布

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.probs(wires=0)

    # 注意：不要從 θ=0 出發！P(1)=sin²(θ/2)，在 θ=0 時梯度 = 0.5·sin(0) = 0，
    # 會卡在鞍點。從非零角度開始梯度才非零。
    theta = pnp.array(0.5, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.3)

    def cost(theta):
        """成本函式：與目標分布 |1⟩ 的平方距離（用 qml.math 做 autograd 分派）。"""
        return qml.math.sum((circuit(theta) - target) ** 2)

    print(f"初始 θ={theta:.4f} → probs={circuit(theta)}")
    for step in range(50):
        theta, cost_val = opt.step_and_cost(cost, theta)
        if step % 10 == 0:
            print(f"step {step:3d}  cost={cost_val:.5f}  θ={theta:.4f}")

    final = circuit(theta)
    print(f"最終 θ={theta:.4f} → probs={final}")
    print("目標 |1⟩ 的機率為", f"{final[1]:.4f}", "（越接近 1 越好）")


def demo_qiskit_vqc():
    """Qiskit：量子變分分類器 (VQC) 做二元分類。"""
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit_machine_learning.algorithms import VQC
    from scipy.optimize import minimize  # 作為 VQC 的 optimizer (Minimizer)

    print("\n2) Qiskit VQC 二元分類")

    # 合成一個可線性分離的資料集
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (60, 2))
    y = np.array([1 if x[0] * x[1] > 0 else 0 for x in X])

    # Qiskit 2.1+ 建議用函式式 API（回傳一般 QuantumCircuit，避免棄用警告）
    feature_map = zz_feature_map(2, reps=2)
    ansatz = real_amplitudes(2, reps=2)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=minimize,
        initial_point=np.zeros(ansatz.num_parameters),
    )
    vqc.fit(X, y)

    acc = vqc.score(X, y)
    print(f"訓練準確率：{acc * 100:.1f}%")
    return vqc


def main():
    demo_pennylane_gradient()
    demo_qiskit_vqc()
    print("\n完成！量子機器學習範例執行成功")


if __name__ == "__main__":
    main()
