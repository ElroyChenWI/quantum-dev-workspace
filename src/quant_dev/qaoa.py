"""QAOA reference workload definitions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .workloads import ExpectationWorkload, qiskit_z_string

Edge = tuple[int, int]


def maxcut_workload(
    n_qubits: int,
    edges: Sequence[Edge],
    p: int = 1,
    initial_params: Sequence[float] | None = None,
) -> ExpectationWorkload:
    """Build a QAOA MaxCut expectation workload.

    The observable is the MaxCut value:

        C = sum_(i,j) (1 - Z_i Z_j) / 2

    The workload maximizes <C>.
    """

    edge_list = [(int(i), int(j)) for i, j in edges]
    if p <= 0:
        raise ValueError("p must be positive")
    for i, j in edge_list:
        if i == j:
            raise ValueError("self-loops are not valid MaxCut edges")
        if i < 0 or j < 0 or i >= n_qubits or j >= n_qubits:
            raise ValueError(f"edge out of range: {(i, j)}")

    n_params = 2 * p
    if initial_params is None:
        initial = np.concatenate([np.full(p, 0.6), np.full(p, 0.4)])
    else:
        initial = np.asarray(initial_params, dtype=float)
        if len(initial) != n_params:
            raise ValueError(f"expected {n_params} initial parameters")

    def circuit_builder(params: np.ndarray):
        from qiskit import QuantumCircuit

        theta = np.asarray(params, dtype=float)
        if len(theta) != n_params:
            raise ValueError(f"expected {n_params} parameters")
        gammas = theta[:p]
        betas = theta[p:]

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for layer in range(p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])
            for i, j in edge_list:
                # RZZ(theta) = exp(-i theta/2 Zi Zj). MaxCut cost evolution
                # exp(-i gamma * (1-ZZ)/2) is equivalent to RZZ(-gamma) up
                # to a global phase.
                qc.rzz(-gamma, i, j)
            for q in range(n_qubits):
                qc.rx(2.0 * beta, q)
        return qc

    def observable_builder():
        from qiskit.quantum_info import SparsePauliOp

        terms: list[tuple[str, float]] = [("I" * n_qubits, 0.5 * len(edge_list))]
        for i, j in edge_list:
            terms.append((qiskit_z_string(n_qubits, i, j), -0.5))
        return SparsePauliOp.from_list(terms)

    return ExpectationWorkload(
        name=f"qaoa_maxcut_n{n_qubits}_p{p}",
        n_qubits=n_qubits,
        n_params=n_params,
        circuit_builder=circuit_builder,
        observable_builder=observable_builder,
        objective_direction="maximize",
        initial_params=initial,
        metadata={"algorithm": "QAOA", "problem": "MaxCut", "edges": edge_list, "p": p},
    )


def brute_force_maxcut(n_qubits: int, edges: Sequence[Edge]) -> tuple[int, list[str]]:
    """Return the exact MaxCut value and optimal bitstrings for small graphs."""

    best = -1
    states: list[str] = []
    for x in range(1 << n_qubits):
        value = 0
        for i, j in edges:
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            value += int(bi != bj)
        bitstring = format(x, f"0{n_qubits}b")[::-1]
        if value > best:
            best = value
            states = [bitstring]
        elif value == best:
            states.append(bitstring)
    return best, states
