"""
00 量子糾纏 vs 不糾纏 — 對照實驗
=================================

用四個電路比較「糾纏」與「不糾纏」的差別：

  糾纏：    Bell 態（2 qubit）、GHZ 態（3 qubit）
  不糾纏：  兩個獨立 qubit、三個獨立 qubit

判斷標準（用「互相資訊 Mutual Information」量化糾纏程度）：
  MI = H(q0) + H(q1) - H(q0, q1)
  - 完全獨立 → MI ≈ 0 bits（各測各的，互不相干）
  - 完全糾纏 → MI ≈ 1 bit （測到其中一個，立刻知道另一個）

執行方式：
    python examples/00_entanglement_vs_independent.py
"""

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ---------------------------------------------------------------------------
# 建立四種電路
# ---------------------------------------------------------------------------
def bell_circuit():
    """H + CNOT：產生 Bell 態（2 qubit 完全糾纏）"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def independent_2q_circuit():
    """兩個獨立的 H gate：完全沒有糾纏（2 qubit）"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def ghz_circuit():
    """H + CNOT 鏈：GHZ 態（3 qubit 完全糾纏）"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def independent_3q_circuit():
    """三個獨立的 H gate：完全沒有糾纏（3 qubit）"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


# ---------------------------------------------------------------------------
# 分析工具
# ---------------------------------------------------------------------------
def mutual_information(counts, n_qubits, shots):
    """從測量結果計算「第一個 qubit」與「其餘 qubit」的互相資訊（bits）。

    把 bitstring 分成「第一位」和「剩下的位」，計算兩群的相關程度。
    """
    # P(bit0) 與 P(rest)
    p_bit = np.zeros(2)
    p_rest = np.zeros(2 ** (n_qubits - 1))
    p_joint = np.zeros((2, 2 ** (n_qubits - 1)))

    for bitstring, count in counts.items():
        prob = count / shots
        b0 = int(bitstring[0])                       # 第一位（q0）
        rest = int(bitstring[1:], 2)                 # 其餘位（q1, q2, ...）
        p_bit[b0] += prob
        p_rest[rest] += prob
        p_joint[b0, rest] += prob

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    return entropy(p_bit) + entropy(p_rest) - entropy(p_joint)


def print_summary(name, qc, shots=1024):
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts(qc)
    n_qubits = qc.num_qubits
    mi = mutual_information(counts, n_qubits, shots)

    # 「全部相同」的機率（第一位 = 其餘每一位）
    same_prob = 0.0
    for bitstring, count in counts.items():
        if all(b == bitstring[0] for b in bitstring):
            same_prob += count / shots

    print(f"\n=== {name} ===")
    print(qc.draw())
    print(f"  測量結果：{dict(sorted(counts.items()))}")
    print(f"  互相資訊 MI = {mi:.3f} bits")
    print(f"  「全部相同」機率 = {same_prob * 100:.1f}%")
    return name, counts, mi, same_prob


def plot_comparison(results):
    import matplotlib

    # 讓 matplotlib 支援中文（Windows 用微軟正黑體）
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt

    names = [r[0] for r in results]
    mis = [r[2] for r in results]
    same = [r[3] * 100 for r in results]

    # 用較短的自訂標籤，避免 X 軸文字重疊
    short_names = {
        "Bell 態（2 qubit 糾纏）": "Bell\n(糾纏)",
        "獨立 2 qubit（不糾纏）": "獨立 2q\n(不糾纏)",
        "GHZ 態（3 qubit 糾纏）": "GHZ\n(糾纏)",
        "獨立 3 qubit（不糾纏）": "獨立 3q\n(不糾纏)",
    }
    display_names = [short_names.get(n, n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左圖：互相資訊
    bars = axes[0].bar(display_names, mis, color=["#2ca02c", "#d62728", "#2ca02c", "#d62728"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("互相資訊 MI (bits)")
    axes[0].set_title("糾纏程度（0 = 獨立, 1 = 完全糾纏）")
    axes[0].set_ylim(0, 1.15)
    axes[0].tick_params(axis="x", labelsize=9)
    for b, v in zip(bars, mis):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")

    # 右圖：全部相同的機率
    bars = axes[1].bar(display_names, same, color=["#2ca02c", "#d62728", "#2ca02c", "#d62728"])
    axes[1].set_ylabel("「全部相同」機率 (%)")
    axes[1].set_title("測量時所有 qubit 是否強迫一致")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", labelsize=9)
    for b, v in zip(bars, same):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    fname = OUT_DIR / "00_entanglement_comparison.png"
    plt.savefig(fname, dpi=120)
    print(f"\n[圖表] 對照圖已儲存：{fname}")
    plt.close()


def main():
    print("=" * 60)
    print("量子糾纏 vs 不糾纏 對照實驗")
    print("=" * 60)

    results = []
    results.append(print_summary("Bell 態（2 qubit 糾纏）", bell_circuit()))
    results.append(print_summary("獨立 2 qubit（不糾纏）", independent_2q_circuit()))
    results.append(print_summary("GHZ 態（3 qubit 糾纏）", ghz_circuit()))
    results.append(print_summary("獨立 3 qubit（不糾纏）", independent_3q_circuit()))

    plot_comparison(results)

    print("\n" + "=" * 60)
    print("怎麼解讀？")
    print("  糾纏態（綠色）：MI ≈ 1 bit、全部相同機率 ≈ 100%")
    print("    → 測到一個，立刻知道其他所有 qubit 的結果（手套配對！）")
    print("  不糾纏（紅色）：MI ≈ 0 bit、全部相同機率 ≈ 50% / 12.5%")
    print("    → 每個 qubit 各自獨立擲硬幣，互不相干")
    print("=" * 60)


if __name__ == "__main__":
    main()
