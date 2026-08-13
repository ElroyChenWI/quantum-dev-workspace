"""One-command smoke demo for the quantum development platform.

The script is intentionally local-first:
  1. profile the current machine,
  2. run a router smoke decision,
  3. execute the QAOA MaxCut reference workload.

IBM cloud profiling is optional and never submits a QPU job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(label: str, args: list[str], required: bool = True) -> bool:
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    print(" ".join(args), flush=True)
    proc = subprocess.run(args, cwd=ROOT, check=False)
    if proc.returncode == 0:
        print(f"[ok] {label}")
        return True
    print(f"[failed] {label}: exit code {proc.returncode}")
    if required:
        raise SystemExit(proc.returncode)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the platform quickstart smoke demo.")
    parser.add_argument("--with-cloud", action="store_true", help="Also query IBM account/backend profile.")
    parser.add_argument("--backend", default="ibm_kingston", help="IBM backend to profile when --with-cloud is set.")
    parser.add_argument("--qaoa-maxiter", type=int, default=120, help="QAOA optimizer iterations for quickstart.")
    args = parser.parse_args()

    py = sys.executable
    print("Quantum Dev Workspace quickstart")
    print(f"python={py}")
    print(f"repo={ROOT}", flush=True)

    _run(
        "1. Local environment profile",
        [py, "quantum_vqe/quantum_env_check.py", "--max-qubits", "12", "--depth", "1"],
    )

    if args.with_cloud:
        _run(
            "2. IBM cloud profile",
            [py, "quantum_vqe/quantum_cloud_check.py", "--backend", args.backend],
            required=False,
        )
    else:
        print("\n[skip] IBM cloud profile. Use --with-cloud to query account/backend status.")

    _run(
        "3. Router smoke test",
        [
            py,
            "quantum_vqe/quantum_router.py",
            "--qubits",
            "8",
            "--depth",
            "1",
            "--accuracy",
            "exact",
            "--time-budget",
            "1",
        ],
    )

    _run(
        "4. QAOA MaxCut reference workload",
        [
            py,
            "quantum_vqe/run_qaoa_maxcut.py",
            "--graph",
            "square",
            "--p",
            "2",
            "--maxiter",
            str(args.qaoa_maxiter),
        ],
    )

    print("\nQuickstart complete.")
    print("Key outputs:")
    print(f"- {ROOT / 'quantum_vqe' / 'outputs' / 'env_profile.md'}")
    print(f"- {ROOT / 'quantum_vqe' / 'outputs' / 'qaoa_maxcut_convergence.png'}")
    if args.with_cloud:
        print(f"- {ROOT / 'quantum_vqe' / 'outputs' / 'cloud_profile.md'}")


if __name__ == "__main__":
    main()
