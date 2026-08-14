"""Variational Quantum Classifier (VQC) reference workload.

VQC is the ML face of the same expectation primitive used by VQE and QAOA:
per input sample x, the circuit is `encode(x) + variational(params)` and the
observable score is `<Z_0>`, mapped to a class label via sign.

The point of this module is not that VQC beats classical classifiers - it does
not, at this scale - but that the expectation workload abstraction generalises
from chemistry (VQE) and combinatorial optimization (QAOA) to ML-style losses
without a new primitive. `VQCWorkload.as_expectation(x)` makes that explicit:
every sample becomes a plain ExpectationWorkload the existing executors and
router can already consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .workloads import ExpectationWorkload, qiskit_z_string


@dataclass(frozen=True)
class VQCWorkload:
    """A per-sample expectation workload for binary classification."""

    name: str
    n_qubits: int
    n_params: int
    n_features: int
    circuit_builder: Callable[[np.ndarray, np.ndarray], Any]
    observable_builder: Callable[[], Any]
    initial_params: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def params_or_default(self, seed: int = 0) -> np.ndarray:
        if self.initial_params is not None:
            return np.asarray(self.initial_params, dtype=float)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 0.1, size=self.n_params)

    def as_expectation(self, x: np.ndarray) -> ExpectationWorkload:
        """Freeze a single sample into a plain ExpectationWorkload.

        This is the bridge that lets router / executors / benchmarks written
        for VQE and QAOA consume a VQC sample without special-casing ML.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.n_features,):
            raise ValueError(f"expected feature vector of shape ({self.n_features},)")
        cb = self.circuit_builder
        return ExpectationWorkload(
            name=f"{self.name}_sample",
            n_qubits=self.n_qubits,
            n_params=self.n_params,
            circuit_builder=lambda p, _x=x_arr: cb(p, _x),
            observable_builder=self.observable_builder,
            initial_params=self.initial_params,
            metadata={**self.metadata, "x": x_arr.tolist()},
        )


def binary_vqc_workload(
    n_qubits: int = 2,
    n_layers: int = 3,
    initial_params: Sequence[float] | None = None,
) -> VQCWorkload:
    """Build a two-class VQC: angle encoding, RY/RZ variational layers, ring entanglement.

    Feature vector length must equal n_qubits; features are angle-encoded via
    RY(pi * x_q). The score observable is Z on qubit 0 in [-1, +1]; predict +1
    for score > 0, -1 otherwise.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if n_layers < 1:
        raise ValueError("n_layers must be positive")

    # Hardware-efficient ansatz with data re-uploading: per layer, re-encode
    # features via RY(pi*x), then a universal single-qubit rotation
    # RX(theta)-RY(theta)-RZ(theta) per qubit, then CNOT ring.
    n_params = 3 * n_qubits * n_layers

    if initial_params is None:
        initial = None
    else:
        initial = np.asarray(initial_params, dtype=float)
        if len(initial) != n_params:
            raise ValueError(f"expected {n_params} initial parameters, got {len(initial)}")

    def circuit_builder(params: np.ndarray, x: np.ndarray):
        from qiskit import QuantumCircuit

        theta = np.asarray(params, dtype=float)
        if len(theta) != n_params:
            raise ValueError(f"expected {n_params} parameters")
        xv = np.asarray(x, dtype=float)
        if xv.shape != (n_qubits,):
            raise ValueError(f"expected feature vector of shape ({n_qubits},)")

        qc = QuantumCircuit(n_qubits)
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qc.ry(float(np.pi * xv[q]), q)
            for q in range(n_qubits):
                qc.rx(float(theta[idx]), q)
                idx += 1
                qc.ry(float(theta[idx]), q)
                idx += 1
                qc.rz(float(theta[idx]), q)
                idx += 1
            if n_qubits >= 2:
                for q in range(n_qubits):
                    qc.cx(q, (q + 1) % n_qubits)
        return qc

    def observable_builder():
        from qiskit.quantum_info import SparsePauliOp

        return SparsePauliOp.from_list([(qiskit_z_string(n_qubits, 0), 1.0)])

    return VQCWorkload(
        name=f"vqc_binary_n{n_qubits}_L{n_layers}",
        n_qubits=n_qubits,
        n_params=n_params,
        n_features=n_qubits,
        circuit_builder=circuit_builder,
        observable_builder=observable_builder,
        initial_params=initial,
        metadata={"algorithm": "VQC", "problem": "binary_classification", "n_layers": n_layers},
    )


def moons_dataset(
    n_samples: int = 200,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-moons, standardised to zero mean and unit variance.

    Returns (X, y) with X shape (n_samples, 2) and y in {-1, +1}. No sklearn
    dependency; the shape matches sklearn.datasets.make_moons closely enough
    for a demo classifier.
    """
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    n2 = n_samples - n1
    t1 = rng.uniform(0.0, np.pi, n1)
    t2 = rng.uniform(0.0, np.pi, n2)
    upper = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    lower = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    X = np.concatenate([upper, lower], axis=0)
    X = X + rng.normal(0.0, noise, size=X.shape)
    y = np.concatenate([np.ones(n1, dtype=int), -np.ones(n2, dtype=int)])
    # Min-max scale to [-1, 1] so RY(pi*x) stays within [-pi, pi] and avoids
    # aliasing (RY has period 2*pi; standardising can push |x|>1 and collapse
    # distinct samples onto the same encoded state).
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = 2.0 * (X - X_min) / (X_max - X_min) - 1.0
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    """Sign-based binary accuracy against labels in {-1, +1}."""
    preds = np.where(np.asarray(scores) > 0.0, 1, -1)
    return float(np.mean(preds == np.asarray(y)))
