"""Utilities for resource-aware quantum workload execution."""

from .executors import (
    ExpectationResult,
    objective_value,
    run_qiskit_expectation,
    run_qiskit_vqc_expectation,
)
from .vqc import VQCWorkload, accuracy, binary_vqc_workload, moons_dataset
from .workloads import ExpectationWorkload

__all__ = [
    "ExpectationResult",
    "ExpectationWorkload",
    "VQCWorkload",
    "accuracy",
    "binary_vqc_workload",
    "moons_dataset",
    "objective_value",
    "run_qiskit_expectation",
    "run_qiskit_vqc_expectation",
]

__version__ = "0.1.0"
