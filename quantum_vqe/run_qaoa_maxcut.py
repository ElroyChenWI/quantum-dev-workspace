"""QAOA MaxCut reference workload for the generic expectation engine.

This script demonstrates that the repository is no longer limited to H2 VQE:
the same expectation primitive and resource-aware router can host an
optimization workload.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_dev.executors import objective_value, run_qiskit_expectation
from quant_dev.qaoa import brute_force_maxcut, maxcut_workload
from quant_dev.router import route

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _default_graph(name: str) -> tuple[int, list[tuple[int, int]]]:
    graphs = {
        "triangle": (3, [(0, 1), (1, 2), (0, 2)]),
        "square": (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        "square_diag": (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    }
    if name not in graphs:
        raise ValueError(f"unknown graph: {name}")
    return graphs[name]


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "expectation", "objective", "params"])
        writer.writeheader()
        writer.writerows(history)


def _plot(path: Path, history: list[dict[str, object]], exact: float) -> None:
    import matplotlib.pyplot as plt

    xs = [int(row["iteration"]) for row in history]
    ys = [float(row["expectation"]) for row in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, color="#2f6f8f", marker="o", label="QAOA expectation")
    plt.axhline(exact, color="#4c956c", linestyle="--", label=f"Exact MaxCut = {exact:.0f}")
    plt.xlabel("Evaluation")
    plt.ylabel("Expected cut value")
    plt.title("QAOA MaxCut reference workload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QAOA MaxCut through the generic expectation engine.")
    parser.add_argument("--graph", choices=["triangle", "square", "square_diag"], default="square")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth.")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--time-budget", type=float, default=5.0)
    parser.add_argument("--accuracy", choices=["exact", "estimate", "hardware"], default="exact")
    parser.add_argument("--allow-ibm", action="store_true")
    args = parser.parse_args()

    n_qubits, edges = _default_graph(args.graph)
    workload = maxcut_workload(n_qubits=n_qubits, edges=edges, p=args.p)
    exact_value, exact_states = brute_force_maxcut(n_qubits, edges)

    decision = route(
        qubits=n_qubits,
        depth=args.p,
        accuracy=args.accuracy,
        time_budget_s=args.time_budget,
        allow_ibm=args.allow_ibm,
    )

    print("=" * 72)
    print("QAOA MaxCut reference workload")
    print("=" * 72)
    print(f"graph={args.graph}, qubits={n_qubits}, edges={edges}, p={args.p}")
    print(f"exact_maxcut={exact_value}, optimal_bitstrings={exact_states}")
    print(f"router_selected={decision.selected_backend or 'none'}")
    print(f"router_decision={decision.explanation}")
    if decision.selected_backend not in {"cpu", "cudaq"}:
        print("This reference script currently executes the generic expectation primitive on Qiskit CPU.")

    history: list[dict[str, object]] = []
    eval_count = 0

    def evaluate(params: np.ndarray) -> float:
        nonlocal eval_count
        result = run_qiskit_expectation(workload, params)
        objective = objective_value(workload, result.value)
        history.append(
            {
                "iteration": eval_count,
                "expectation": result.value,
                "objective": objective,
                "params": " ".join(f"{x:.10f}" for x in np.asarray(params, dtype=float)),
            }
        )
        eval_count += 1
        return objective

    initial = workload.params_or_default()
    print(f"initial_params={initial.tolist()}")
    print(f"initial_expected_cut={-evaluate(initial):.6f}")

    result = minimize(
        evaluate,
        initial,
        method="COBYLA",
        options={"maxiter": args.maxiter, "tol": 1e-6, "rhobeg": 0.6},
    )
    best_expected_cut = -float(result.fun)

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "qaoa_maxcut_history.csv"
    png_path = OUT_DIR / "qaoa_maxcut_convergence.png"
    _write_history(csv_path, history)
    _plot(png_path, history, float(exact_value))

    print()
    print(f"completed_evaluations={len(history)}")
    print(f"best_expected_cut={best_expected_cut:.6f}")
    print(f"exact_maxcut={exact_value}")
    print(f"approximation_ratio={best_expected_cut / exact_value:.6f}")
    print(f"history_csv={csv_path}")
    print(f"plot_png={png_path}")


if __name__ == "__main__":
    main()
