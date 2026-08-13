"""Workload abstractions for expectation-based NISQ experiments.

The core primitive is intentionally small:

    circuit(params) + observable -> expectation value

VQE, QAOA, VQC-style losses, and many simulation observables can all be mapped
to this shape while keeping routing and execution policy separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

ObjectiveDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class ExpectationWorkload:
    """A backend-neutral expectation-value workload."""

    name: str
    n_qubits: int
    n_params: int
    circuit_builder: Callable[[np.ndarray], Any]
    observable_builder: Callable[[], Any]
    objective_direction: ObjectiveDirection = "minimize"
    initial_params: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def params_or_default(self) -> np.ndarray:
        if self.initial_params is not None:
            return np.asarray(self.initial_params, dtype=float)
        return np.zeros(self.n_params, dtype=float)


def qiskit_z_string(n_qubits: int, *qubits: int) -> str:
    """Return a Qiskit Pauli label with Z on the requested logical qubits.

    Qiskit Pauli labels are little-endian with respect to circuit qubit indices:
    the rightmost character acts on qubit 0.
    """

    chars = ["I"] * n_qubits
    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"qubit index out of range: {qubit}")
        chars[n_qubits - 1 - qubit] = "Z"
    return "".join(chars)
