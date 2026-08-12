"""
H2 分子的 2-qubit 哈密頓量（共用定義）
=======================================

這是 Qiskit / CUDA-Q 官方 VQE 範例採用的標準 H2 哈密頓量
（STO-3G 基底、R = 0.735 Å、parity 對稱性簡化到 2 qubits）。

單位：Hartree（哈特里）。

三個框架（Qiskit / PennyLane / CUDA-Q）都讀取這一份定義，
確保「同一個物理問題」→ 收斂到「同一個基態能量」。
"""

from __future__ import annotations

# Pauli 字串（順序：q0, q1）+ 係數
#   I = 單位矩陣, X/Y/Z = Pauli 矩陣
PAULI_TERMS: list[tuple[str, float]] = [
    ("II", -1.052373245772859),
    ("ZI",  0.39793742484318045),
    ("IZ", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX",  0.18093119978423156),
]

# 建議的初始參數（小幅隨機，避開退化點）
INITIAL_PARAMS = [0.10, 0.10, 0.05, 0.05]


def exact_ground_state_energy() -> float:
    """用 numpy 直接對角化哈密頓量矩陣，回傳精確基態能量（作為驗證基準）。"""
    import numpy as np

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {"I": I, "X": X, "Y": Y, "Z": Z}

    H = np.zeros((4, 4), dtype=complex)
    for pauli, coef in PAULI_TERMS:
        # pauli 字串 e.g. "ZI" → Z 作用在 q0, I 作用在 q1
        # numpy kron：左邊是較高位元（q0），右邊是較低位元（q1）
        H += coef * np.kron(pauli_map[pauli[0]], pauli_map[pauli[1]])

    energies = np.linalg.eigvalsh(H)
    return float(energies[0])


def describe():
    """回傳人類可讀的說明。"""
    lines = ["H2 哈密頓量（2 qubits）:"]
    for pauli, coef in PAULI_TERMS:
        lines.append(f"  {coef:+.4f} * {pauli}")
    lines.append(f"  精確基態能量 = {exact_ground_state_energy():.6f} Ha")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
