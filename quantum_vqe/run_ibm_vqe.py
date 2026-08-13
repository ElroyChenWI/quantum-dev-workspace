"""
Hardware-in-the-loop VQE on IBM Quantum.

This script runs a small VQE loop where the classical optimizer runs locally
and each energy evaluation is submitted to IBM Quantum Runtime Estimator.
It is intentionally separate from run_ibm_cloud.py, which performs only one
fixed-parameter Estimator evaluation.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2_hamiltonian import INITIAL_PARAMS, exact_ground_state_energy
from run_ibm_cloud import build_h2_ansatz, build_h2_observable, load_token

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small IBM hardware-in-the-loop VQE.")
    parser.add_argument("--backend", default="ibm_kingston", help="IBM backend name.")
    parser.add_argument("--maxiter", type=int, default=12, help="Maximum optimizer evaluations.")
    parser.add_argument("--outfile", default="ibm_vqe_history.csv", help="Output CSV filename.")
    args = parser.parse_args()

    token = load_token()
    if not token:
        raise SystemExit("Missing QISKIT_IBM_TOKEN in .env or environment.")

    from qiskit.primitives import StatevectorEstimator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)
    status = backend.status()
    print(f"backend={backend.name} qubits={backend.num_qubits} pending_jobs={status.pending_jobs}")

    ansatz = build_h2_ansatz()
    observable = build_h2_observable()
    exact = exact_ground_state_energy()

    local_initial = StatevectorEstimator().run(
        [(ansatz, observable, [INITIAL_PARAMS])]
    ).result()[0].data.evs[0]
    print(f"exact_ground_state={exact:.9f} Ha")
    print(f"local_initial_expectation={float(local_initial):.9f} Ha")
    print(f"initial_params={INITIAL_PARAMS}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(ansatz)
    isa_observable = observable.apply_layout(isa_circuit.layout)
    estimator = Estimator(backend)

    history: list[dict[str, object]] = []
    out_path = OUT_DIR / args.outfile
    OUT_DIR.mkdir(exist_ok=True)

    def write_history() -> None:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["eval", "job_id", "energy", "elapsed_s", "params"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def energy(params: np.ndarray) -> float:
        eval_no = len(history) + 1
        p = [float(x) for x in params]
        print(f"\n[{eval_no}/{args.maxiter}] submit theta={np.array2string(np.array(p), precision=5)}", flush=True)
        t0 = time.perf_counter()
        job = estimator.run([(isa_circuit, isa_observable, [p])])
        print(f"  job_id={job.job_id()}", flush=True)
        result = job.result()
        elapsed = time.perf_counter() - t0
        value = float(np.atleast_1d(np.asarray(result[0].data.evs))[0])
        history.append(
            {
                "eval": eval_no,
                "job_id": job.job_id(),
                "energy": f"{value:.12f}",
                "elapsed_s": f"{elapsed:.3f}",
                "params": " ".join(f"{x:.12g}" for x in p),
            }
        )
        write_history()
        print(f"  energy={value:.9f} Ha elapsed={elapsed:.1f}s", flush=True)
        return value

    result = minimize(
        energy,
        np.array(INITIAL_PARAMS, dtype=float),
        method="COBYLA",
        options={"maxiter": args.maxiter, "rhobeg": 0.25, "tol": 1e-3},
    )

    print("\nIBM hardware-in-the-loop VQE complete")
    print(f"evaluations={len(history)}")
    print(f"best_energy={float(result.fun):.9f} Ha")
    print(f"best_params={np.array2string(np.asarray(result.x), precision=6)}")
    print(f"history_csv={out_path}")
    print("Note: this is a small noisy-hardware VQE run; results depend on shots, queue, calibration, and noise.")


if __name__ == "__main__":
    main()
