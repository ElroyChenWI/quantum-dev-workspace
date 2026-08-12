"""常用量子電路輔助函式。

這個模組提供一些小工具，方便在筆記本中重複使用：
- 產生 GHZ 態、Bell 態等經典電路
- 從測量結果繪製機率分布
- 同時支援 Qiskit 與 PennyLane 兩種後端
"""

from __future__ import annotations

import numpy as np


def bell_state_circuit_qiskit():
    """建立 Qiskit 的 Bell 態 |Φ⁺⟩ 電路（2 qubits）。"""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def ghz_state_circuit_qiskit(n_qubits: int = 3):
    """建立 Qiskit 的 GHZ 態電路：(|0...0⟩ + |1...1⟩)/√2。"""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def probability_from_counts(counts: dict, n_shots: int) -> dict:
    """把 Qiskit 的 counts 轉成機率字典。"""
    total = sum(counts.values()) if n_shots is None else n_shots
    return {k: v / total for k, v in counts.items()}


def plot_probabilities(probs: dict, title: str = "Measurement Probabilities"):
    """繪製測量結果的機率長條圖。"""
    import matplotlib.pyplot as plt

    states = sorted(probs.keys())
    values = [probs[s] for s in states]
    plt.figure(figsize=(max(6, len(states) * 0.6), 4))
    plt.bar(states, values)
    plt.xlabel("Basis state")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1.05)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    return plt


def demo_circuit_pennylane(n_qubits: int = 2):
    """建立一個 PennyLane 的可微分量子函式（示範用）。"""
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit(weights):
        for w in range(n_qubits):
            qml.RY(weights[w], wires=w)
        qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")
        return qml.probs(wires=range(n_qubits))

    return circuit
