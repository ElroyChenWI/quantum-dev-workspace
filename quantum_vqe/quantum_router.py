"""CLI for the resource-aware quantum execution router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_dev.router import route


def _fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def _fmt_gb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a variational quantum workload to a backend.")
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--accuracy", choices=["exact", "hardware", "estimate"], default="exact")
    parser.add_argument("--time-budget", type=float, default=60.0, help="Runtime budget per expectation in seconds.")
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--vram-gb", type=float, default=6.0)
    parser.add_argument("--allow-ibm", action="store_true", help="Allow IBM QPU to be selected/recommended.")
    args = parser.parse_args()

    decision = route(
        qubits=args.qubits,
        depth=args.depth,
        accuracy=args.accuracy,
        time_budget_s=args.time_budget,
        ram_gb=args.ram_gb,
        vram_gb=args.vram_gb,
        allow_ibm=args.allow_ibm,
    )

    print("Resource-aware quantum routing")
    print("=" * 40)
    print(f"problem: qubits={args.qubits}, depth={args.depth}, accuracy={args.accuracy}")
    print(f"budgets: time<={args.time_budget:.1f}s, RAM<={args.ram_gb:.1f}GB, VRAM<={args.vram_gb:.1f}GB")
    print()
    print("| backend | available | feasible | est_time | est_memory | reason |")
    print("|---|---:|---:|---:|---:|---|")
    for est in decision.estimates:
        print(
            f"| {est.backend} | {est.available} | {est.feasible} | "
            f"{_fmt_seconds(est.estimated_time_s)} | {_fmt_gb(est.estimated_memory_gb)} | {est.reason} |"
        )
    print()
    print(f"selected_backend: {decision.selected_backend or 'none'}")
    print(f"decision: {decision.explanation}")


if __name__ == "__main__":
    main()
