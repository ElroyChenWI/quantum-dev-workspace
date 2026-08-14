"""Execution primitives shared by reference workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .workloads import ExpectationWorkload
from .vqc import VQCWorkload


@dataclass(frozen=True)
class ExpectationResult:
    workload: str
    backend: str
    value: float
    metadata: dict[str, Any]


def run_qiskit_expectation(workload: ExpectationWorkload, params: np.ndarray) -> ExpectationResult:
    """Evaluate an expectation workload with Qiskit StatevectorEstimator."""

    from qiskit.primitives import StatevectorEstimator

    theta = np.asarray(params, dtype=float)
    circuit = workload.circuit_builder(theta)
    observable = workload.observable_builder()
    estimator = StatevectorEstimator()
    result = estimator.run([(circuit, observable)]).result()
    value = float(np.atleast_1d(np.asarray(result[0].data.evs))[0])
    return ExpectationResult(
        workload=workload.name,
        backend="qiskit_statevector",
        value=value,
        metadata={
            "n_qubits": workload.n_qubits,
            "n_params": workload.n_params,
            "objective_direction": workload.objective_direction,
        },
    )


def run_qiskit_vqc_expectation(
    workload: VQCWorkload,
    params: np.ndarray,
    x: np.ndarray,
) -> ExpectationResult:
    """Evaluate a single VQC sample by freezing it into an ExpectationWorkload.

    This is the concrete proof that VQC reuses the same expectation primitive
    as VQE and QAOA: it goes through the exact same executor path.
    """
    frozen = workload.as_expectation(x)
    return run_qiskit_expectation(frozen, params)


def objective_value(workload: ExpectationWorkload, expectation: float) -> float:
    """Convert an expectation into a minimization objective."""

    if workload.objective_direction == "minimize":
        return expectation
    if workload.objective_direction == "maximize":
        return -expectation
    raise ValueError(f"unknown objective direction: {workload.objective_direction}")
