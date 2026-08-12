"""
03 量子模擬 — 演化與物理系統
============================

內容：
1. 用 Qiskit 模擬量子系統的時間演化（Trotter 分解）
2. 用 Cirq 檢視 GHZ 態的狀態向量與密度矩陣

執行方式：
    python examples/03_quantum_simulation.py
"""

import numpy as np


def demo_trotter_evolution():
    """Qiskit：用 Trotter 分解模擬橫場 Ising 模型的時間演化。

    H = X₀X₁ + Z₀Z₁ + Z₀ + Z₁  （非對易項，所以 Trotter 有誤差）

    對比「步數越少 vs 越多」的 fidelity，觀察 Trotter 逼近的收斂性。
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    print("=" * 60)
    print("1) 用 Trotter 分解模擬 H = X₀X₁ + Z₀Z₁ + Z₀ + Z₁ 的演化")

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    # 哈密頓量矩陣（Qiskit 慣例：qubit 0 是 LSB）
    H_mat = np.kron(X, X) + np.kron(Z, Z) + np.kron(I, Z) + np.kron(Z, I)

    t_total = 0.5
    psi0 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩

    # 精確解（用指數矩陣直接算）作為對照
    from scipy.linalg import expm
    exact = expm(-1j * H_mat * t_total) @ psi0

    def trotter_circuit(n_steps):
        """建 n_steps 步的一階 Trotter 電路。"""
        dt = t_total / n_steps
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        for _ in range(n_steps):
            # e^{-i X₀X₁ Δt}：CNOT → Rx(control) → CNOT
            qc.cx(qr[0], qr[1]); qc.rx(2 * dt, qr[0]); qc.cx(qr[0], qr[1])
            # e^{-i Z₀Z₁ Δt}：CNOT → Rz(target) → CNOT
            qc.cx(qr[0], qr[1]); qc.rz(2 * dt, qr[1]); qc.cx(qr[0], qr[1])
            # e^{-i Z₀ Δt}、e^{-i Z₁ Δt}：直接 Rz
            qc.rz(2 * dt, qr[0])
            qc.rz(2 * dt, qr[1])
        return qc

    print("步數越多 → Trotter 逼近越準確（fidelity 越接近 1）")
    for n_steps in (1, 5, 20):
        qc = trotter_circuit(n_steps)
        psi = Statevector(qc).data
        fidelity = abs(np.vdot(exact, psi)) ** 2
        print(f"  n_steps = {n_steps:3d} → fidelity = {fidelity:.5f}")


def demo_cirq_statevector():
    """Cirq：檢視 GHZ 態的狀態向量與密度矩陣。"""
    import cirq

    print("\n2) 用 Cirq 檢視 GHZ 態的狀態向量")

    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[2]),
    )
    sim = cirq.Simulator()
    state = sim.simulate(circuit)

    print("狀態向量：")
    print(state.final_state_vector)
    print("\n密度矩陣（partial trace 前的完整 density matrix）：")
    dm = cirq.density_matrix_from_state_vector(state.final_state_vector)
    print(np.round(dm, 4))

    # 單一 qubit 的 reduced density matrix（顯示與環境的糾纏）
    rho0 = cirq.density_matrix_from_state_vector(state.final_state_vector, indices=[0])
    print("\nqubit 0 的 reduced density matrix：")
    print(np.round(rho0, 4))
    print("（對角元素相等表示 qubit 0 處於最大混合態 → 與其他 qubit 完全糾纏）")


def main():
    demo_trotter_evolution()
    demo_cirq_statevector()
    print("\n完成！量子模擬範例執行成功")


if __name__ == "__main__":
    main()
