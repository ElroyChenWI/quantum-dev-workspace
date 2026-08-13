"""
Batched hardware-in-the-loop VQE for IBM Quantum.

This script is designed for real QPU runs where one-job-per-energy-evaluation
is too slow. Each optimization round submits a batch of candidate parameter
vectors in one Estimator job, then performs a simple pattern-search update
locally. A Runtime Session is used by default to reduce queue overhead between
rounds.

It is still a noisy-hardware demonstration, not a production chemistry solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2_hamiltonian import INITIAL_PARAMS, exact_ground_state_energy
from run_ibm_cloud import build_h2_ansatz, build_h2_observable, load_token

OUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class OptimizerState:
    round: int
    current_energy: float
    current_params: list[float]
    best_energy: float
    best_params: list[float]
    step: float
    no_improve_rounds: int


def _params_to_text(params: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(x):.12g}" for x in params)


def _candidate_batch(theta: np.ndarray, step: float) -> tuple[list[str], np.ndarray]:
    """Return current point plus +/- coordinate perturbations."""
    labels = ["current"]
    candidates = [theta.copy()]
    for i in range(theta.size):
        plus = theta.copy()
        plus[i] += step
        labels.append(f"p{i}+")
        candidates.append(plus)

        minus = theta.copy()
        minus[i] -= step
        labels.append(f"p{i}-")
        candidates.append(minus)
    return labels, np.asarray(candidates, dtype=float)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    fieldnames = [
        "round",
        "candidate",
        "job_id",
        "energy",
        "is_round_best",
        "accepted",
        "step",
        "elapsed_s",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_state(path: Path, state: OptimizerState) -> None:
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _load_state(path: Path) -> OptimizerState:
    data = json.loads(path.read_text(encoding="utf-8"))
    return OptimizerState(**data)


def _plot(history_csv: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    rounds: list[int] = []
    best_by_round: list[float] = []
    current_best = float("inf")
    with history_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_round_best"] != "1":
                continue
            r = int(row["round"])
            e = float(row["energy"])
            current_best = min(current_best, e)
            rounds.append(r)
            best_by_round.append(current_best)

    if not rounds:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rounds, best_by_round, "o-", color="#9467bd", label="best-so-far IBM energy")
    ax.axhline(exact_ground_state_energy(), color="#1f77b4", linestyle=":", label="exact ground state")
    ax.set_xlabel("Batched optimization round")
    ax.set_ylabel("Energy / expectation (Ha)")
    ax.set_title("IBM batched hardware-in-the-loop VQE")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batched hardware-in-the-loop VQE on IBM Quantum.")
    parser.add_argument("--backend", default="ibm_kingston", help="IBM backend name.")
    parser.add_argument("--rounds", type=int, default=20, help="Maximum batched optimization rounds.")
    parser.add_argument("--initial-step", type=float, default=0.45, help="Initial pattern-search step size.")
    parser.add_argument("--min-step", type=float, default=0.03, help="Stop once step size drops below this value.")
    parser.add_argument("--shrink", type=float, default=0.5, help="Step shrink factor after no improvement.")
    parser.add_argument("--tol", type=float, default=0.015, help="Required energy improvement to accept a move.")
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many non-improving rounds.")
    parser.add_argument("--session-max-time", default="2h", help="IBM Runtime Session max_time.")
    parser.add_argument("--outfile", default="ibm_vqe_batched_history.csv")
    parser.add_argument("--statefile", default="ibm_vqe_batched_state.json")
    parser.add_argument("--resume", action="store_true", help="Resume from statefile if present.")
    parser.add_argument("--no-session", action="store_true", help="Run without an IBM Runtime Session.")
    args = parser.parse_args()

    token = load_token()
    if not token:
        raise SystemExit("Missing QISKIT_IBM_TOKEN in .env or environment.")

    from qiskit.primitives import StatevectorEstimator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import QiskitRuntimeService, Session

    OUT_DIR.mkdir(exist_ok=True)
    history_path = OUT_DIR / args.outfile
    state_path = OUT_DIR / args.statefile
    plot_path = OUT_DIR / "ibm_vqe_batched_trajectory.png"

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend)
    status = backend.status()
    print(f"backend={backend.name} qubits={backend.num_qubits} pending_jobs={status.pending_jobs}", flush=True)

    ansatz = build_h2_ansatz()
    observable = build_h2_observable()

    local_initial = StatevectorEstimator().run(
        [(ansatz, observable, [INITIAL_PARAMS])]
    ).result()[0].data.evs[0]
    exact = exact_ground_state_energy()
    print(f"exact_ground_state={exact:.9f} Ha", flush=True)
    print(f"local_initial_expectation={float(local_initial):.9f} Ha", flush=True)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(ansatz)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    if args.resume and state_path.exists():
        state = _load_state(state_path)
        theta = np.asarray(state.current_params, dtype=float)
        best_theta = np.asarray(state.best_params, dtype=float)
        step = float(state.step)
        start_round = int(state.round) + 1
        best_energy = float(state.best_energy)
        current_energy = float(state.current_energy)
        no_improve = int(state.no_improve_rounds)
        rows: list[dict[str, object]] = []
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        print(f"resuming from round={state.round} best={best_energy:.9f} step={step:.4f}", flush=True)
    else:
        theta = np.asarray(INITIAL_PARAMS, dtype=float)
        best_theta = theta.copy()
        step = float(args.initial_step)
        start_round = 1
        best_energy = float("inf")
        current_energy = float("inf")
        no_improve = 0
        rows = []

    mode = backend
    session = None
    if not args.no_session:
        session = Session(backend=backend, max_time=args.session_max_time)
        mode = session
        print(f"session_started max_time={args.session_max_time}", flush=True)

    try:
        estimator = Estimator(mode=mode)
        for round_no in range(start_round, args.rounds + 1):
            if step < args.min_step:
                print(f"stop: step {step:.5f} < min_step {args.min_step:.5f}", flush=True)
                break
            if no_improve >= args.patience:
                print(f"stop: no improvement for {no_improve} rounds", flush=True)
                break

            labels, candidates = _candidate_batch(theta, step)
            print(
                f"\n[round {round_no}/{args.rounds}] submit {len(candidates)} candidates "
                f"step={step:.5f} theta={np.array2string(theta, precision=5)}",
                flush=True,
            )
            t0 = time.perf_counter()
            job = estimator.run([(isa_circuit, isa_observable, candidates.tolist())])
            print(f"  job_id={job.job_id()}", flush=True)
            result = job.result()
            elapsed = time.perf_counter() - t0
            energies = np.asarray(result[0].data.evs, dtype=float).reshape(-1)

            best_idx = int(np.argmin(energies))
            round_best_energy = float(energies[best_idx])
            round_best_theta = candidates[best_idx].copy()
            accepted = round_best_energy < best_energy - args.tol
            if accepted:
                theta = round_best_theta
                best_theta = round_best_theta.copy()
                best_energy = round_best_energy
                current_energy = round_best_energy
                no_improve = 0
            else:
                step *= args.shrink
                no_improve += 1

            for i, (label, params, energy) in enumerate(zip(labels, candidates, energies)):
                rows.append(
                    {
                        "round": round_no,
                        "candidate": label,
                        "job_id": job.job_id(),
                        "energy": f"{float(energy):.12f}",
                        "is_round_best": "1" if i == best_idx else "0",
                        "accepted": "1" if accepted and i == best_idx else "0",
                        "step": f"{step:.12g}",
                        "elapsed_s": f"{elapsed:.3f}",
                        "params": _params_to_text(params),
                    }
                )

            state = OptimizerState(
                round=round_no,
                current_energy=current_energy,
                current_params=[float(x) for x in theta],
                best_energy=best_energy,
                best_params=[float(x) for x in best_theta],
                step=step,
                no_improve_rounds=no_improve,
            )
            _write_rows(history_path, rows)
            _write_state(state_path, state)
            _plot(history_path, plot_path)

            print(
                f"  round_best={round_best_energy:.9f} label={labels[best_idx]} "
                f"accepted={accepted} global_best={best_energy:.9f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        print("\nIBM batched hardware-in-the-loop VQE complete", flush=True)
        print(f"best_energy={best_energy:.9f} Ha", flush=True)
        print(f"best_params={np.array2string(best_theta, precision=6)}", flush=True)
        print(f"history_csv={history_path}", flush=True)
        print(f"state_json={state_path}", flush=True)
        print(f"plot={plot_path}", flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
