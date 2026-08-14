"""VQC reference workload: gradient-trained binary classifier on two moons.

This script demonstrates the ML face of the shared expectation primitive.
Training uses PennyLane's autograd interface (Adam), which mirrors a standard
PyTorch-style training loop; the final parameters are then validated on the
Qiskit statevector executor via `VQCWorkload.as_expectation(x)`, showing that
the same primitive underlying H2 VQE and QAOA MaxCut also drives VQC.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_dev.executors import run_qiskit_vqc_expectation
from quant_dev.vqc import accuracy, binary_vqc_workload, moons_dataset

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _pennylane_circuit(n_qubits: int, n_layers: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params, x):
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RY(np.pi * x[q], wires=q)
            for q in range(n_qubits):
                qml.RX(params[idx], wires=q)
                idx += 1
                qml.RY(params[idx], wires=q)
                idx += 1
                qml.RZ(params[idx], wires=q)
                idx += 1
            if n_qubits >= 2:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return circuit


def _train(
    circuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_params: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    import pennylane as qml
    from pennylane import numpy as pnp

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.normal(0.0, 0.1, size=n_params), requires_grad=True)
    y_pnp = pnp.array(y_train, requires_grad=False)

    def loss_fn(p):
        preds = pnp.stack([circuit(p, x) for x in X_train])
        return pnp.mean((preds - y_pnp) ** 2)

    opt = qml.AdamOptimizer(stepsize=lr)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        params, loss_val = opt.step_and_cost(loss_fn, params)
        scores = np.array([float(circuit(params, x)) for x in X_train])
        acc = accuracy(scores, y_train)
        history.append({"epoch": epoch, "loss": float(loss_val), "train_accuracy": acc})
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  epoch={epoch:03d} loss={float(loss_val):.4f} train_acc={acc:.3f}")
    return np.array(params, dtype=float), history


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_accuracy"])
        writer.writeheader()
        writer.writerows(history)


def _plot_loss(path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    accs = [row["train_accuracy"] for row in history]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, losses, color="#2f6f8f", label="MSE loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#2f6f8f")
    ax1.tick_params(axis="y", labelcolor="#2f6f8f")

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color="#4c956c", label="Train accuracy")
    ax2.set_ylabel("Accuracy", color="#4c956c")
    ax2.set_ylim(0.0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#4c956c")

    plt.title("VQC training on two moons")
    fig.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def _plot_decision_boundary(
    path: Path,
    circuit,
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    grid: int = 60,
) -> None:
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros_like(XX)
    for i in range(grid):
        for j in range(grid):
            Z[i, j] = float(circuit(params, np.array([XX[i, j], YY[i, j]])))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(XX, YY, Z, levels=20, cmap="RdBu", alpha=0.75, vmin=-1, vmax=1)
    ax.contour(XX, YY, Z, levels=[0.0], colors="k", linewidths=1.2)
    pos = y == 1
    ax.scatter(X[pos, 0], X[pos, 1], c="#b71c1c", edgecolor="white", s=28, label="class +1")
    ax.scatter(X[~pos, 0], X[~pos, 1], c="#0d47a1", edgecolor="white", s=28, label="class -1")
    ax.set_title("VQC decision boundary (<Z_0> sign)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def _validate_qiskit(workload, params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Re-evaluate every sample on the Qiskit executor for framework parity."""
    scores = np.zeros(len(X), dtype=float)
    for i, x in enumerate(X):
        result = run_qiskit_vqc_expectation(workload, params, x)
        scores[i] = result.value
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VQC on two moons and validate on Qiskit.")
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--validate-samples", type=int, default=24,
                        help="How many samples to re-score on Qiskit for parity check.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.n_qubits != 2:
        raise SystemExit("The moons dataset is 2D; use --n-qubits 2 for the default demo.")

    print("=" * 72)
    print("VQC reference workload — two moons")
    print("=" * 72)

    X, y = moons_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)
    n_test = max(1, int(len(X) * args.test_frac))
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]
    print(f"train={len(X_train)}, test={len(X_test)}, class_balance_train={float(np.mean(y_train==1)):.2f}")

    workload = binary_vqc_workload(n_qubits=args.n_qubits, n_layers=args.n_layers)
    print(f"workload={workload.name}, n_params={workload.n_params}")

    circuit = _pennylane_circuit(args.n_qubits, args.n_layers)

    print(f"training with PennyLane autograd + Adam (lr={args.lr}, epochs={args.epochs})")
    t0 = time.perf_counter()
    params, history = _train(circuit, X_train, y_train, workload.n_params,
                             args.epochs, args.lr, args.seed)
    train_time = time.perf_counter() - t0
    print(f"training_seconds={train_time:.2f}")

    train_scores = np.array([float(circuit(params, x)) for x in X_train])
    test_scores = np.array([float(circuit(params, x)) for x in X_test])
    train_acc = accuracy(train_scores, y_train)
    test_acc = accuracy(test_scores, y_test)
    print(f"final_train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    n_val = min(args.validate_samples, len(X_test))
    if n_val > 0:
        qiskit_scores = _validate_qiskit(workload, params, X_test[:n_val])
        pl_scores = test_scores[:n_val]
        max_abs_diff = float(np.max(np.abs(pl_scores - qiskit_scores)))
        sign_agreement = float(np.mean(np.sign(pl_scores) == np.sign(qiskit_scores)))
        print(f"framework_parity (PennyLane vs Qiskit statevector) on {n_val} test samples:")
        print(f"  max_abs_score_diff={max_abs_diff:.2e}, sign_agreement={sign_agreement:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "vqc_training_history.csv"
    _write_history(csv_path, history)
    print(f"history_csv={csv_path}")

    if not args.no_plot:
        loss_png = OUT_DIR / "vqc_training_curve.png"
        bd_png = OUT_DIR / "vqc_decision_boundary.png"
        _plot_loss(loss_png, history)
        _plot_decision_boundary(bd_png, circuit, params, X, y)
        print(f"loss_plot_png={loss_png}")
        print(f"decision_boundary_png={bd_png}")


if __name__ == "__main__":
    main()
