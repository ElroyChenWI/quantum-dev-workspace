"""
01 量子計算入門 — Qiskit 與 Cirq 基礎
=====================================

內容：
1. 單量子位元：疊加態與測量
2. Bell 態（糾纏態）
3. GHZ 態（三量子位元糾纏）
4. 用 Qiskit Aer 模擬器執行並繪製機率分布
5. 用 Cirq 建同樣的電路做對照

執行方式：
    python examples/01_quantum_computing_basics.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. 用 Qiskit 建立 Bell 態 |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit


def build_bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)          # 對 qubit 0 做 Hadamard → 疊加態
    qc.cx(0, 1)      # CNOT 產生糾纏
    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# 2. GHZ 態：(|000⟩ + |111⟩)/√2
# ---------------------------------------------------------------------------
def build_ghz_circuit(n=3):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def run_qiskit(qc, shots=1024):
    """用 Aer 模擬器執行並回傳 counts 與機率。"""
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    probs = {k: v / shots for k, v in counts.items()}
    return counts, probs


def plot_probs(probs, title):
    from pathlib import Path

    import matplotlib.pyplot as plt

    states = sorted(probs.keys())
    values = [probs[s] for s in states]
    plt.figure(figsize=(max(5, len(states) * 0.8), 4))
    bars = plt.bar(states, values)
    # 用綠色標記非零機率，直觀看到疊加/糾纏的效果
    for b, v in zip(bars, values):
        b.set_color("#2ca02c" if v > 1e-6 else "#d0d0d0")
    plt.xlabel("Basis state")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    # 儲存到 outputs/ 資料夾（避免 plt.show() 在腳本模式阻塞）
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"{title.replace(' ', '_').replace('|', '').replace('⟩', '')}.png"
    plt.savefig(fname, dpi=120)
    print(f"  圖表已儲存：{fname}")
    plt.close()


def demo_cirq():
    """用 Cirq 建立同樣的 Bell 電路做對照。"""
    import cirq

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m"),
    )
    print("\n--- Cirq Bell circuit ---")
    print(circuit)

    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=1000)
    print("Counts:", result.histogram(key="m"))


def main():
    print("=" * 60)
    print("Qiskit 版本：", end="")
    import qiskit
    print(qiskit.__version__)

    # --- Bell 態 ---
    print("\n--- Qiskit Bell circuit ---")
    bell = build_bell_circuit()
    print(bell.draw())
    counts, probs = run_qiskit(bell)
    print("Counts:", counts)
    plot_probs(probs, "Bell state |Φ⁺⟩")

    # --- GHZ 態 ---
    print("\n--- Qiskit GHZ circuit (3 qubits) ---")
    ghz = build_ghz_circuit(3)
    counts, probs = run_qiskit(ghz)
    print("Counts:", counts)
    plot_probs(probs, "GHZ state (3 qubits)")

    # --- Cirq 對照 ---
    demo_cirq()

    print("\n完成！你已經成功執行量子電路模擬")


if __name__ == "__main__":
    main()
