"""Resource-aware backend routing for variational quantum workflows.

The router uses benchmark CSV files produced by this repository to estimate
runtime and memory requirements, then recommends an execution backend.
It is intentionally conservative: IBM Quantum is recommended only for
hardware-behavior requests or when local simulation is infeasible.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


BYTES_PER_COMPLEX128 = 16


@dataclass(frozen=True)
class BackendEstimate:
    backend: str
    available: bool
    feasible: bool
    estimated_time_s: float | None
    estimated_memory_gb: float | None
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    selected_backend: str | None
    estimates: list[BackendEstimate]
    explanation: str


def statevector_memory_gb(qubits: int) -> float:
    return (2**qubits * BYTES_PER_COMPLEX128) / (1024**3)


def _read_stack_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "qubits": float(row["qubits"]),
                        "depth": float(row["depth"]),
                        "runtime_s": float(row["runtime_s"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _fit_log_runtime(rows: list[dict[str, float]]) -> tuple[float, float, float] | None:
    """Fit log(time) = a + b*qubits + c*log(depth).

    Uses a small normal-equation least-squares solve implemented without
    adding a heavier dependency. Returns None if there is insufficient data.
    """
    valid = [r for r in rows if r["runtime_s"] > 0 and r["depth"] > 0 and r["qubits"] > 0]
    if len(valid) < 3:
        return None

    # Solve X^T X beta = X^T y for beta=(a,b,c).
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for r in valid:
        x = [1.0, r["qubits"], math.log(r["depth"])]
        y = math.log(r["runtime_s"])
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return _solve_3x3(xtx, xty)


def _solve_3x3(a: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    # Gaussian elimination with partial pivoting.
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return (m[0][3], m[1][3], m[2][3])


def _predict_runtime(model: tuple[float, float, float] | None, qubits: int, depth: int) -> float | None:
    if model is None:
        return None
    a, b, c = model
    return math.exp(a + b * qubits + c * math.log(max(depth, 1)))


def _availability() -> dict[str, bool]:
    available = {"cpu": False, "cudaq": False, "ibm": False}
    try:
        import qiskit  # noqa: F401

        available["cpu"] = True
    except ImportError:
        pass
    try:
        import cudaq  # noqa: F401

        available["cudaq"] = True
    except ImportError:
        pass
    # Treat IBM as configured if a token source exists; do not validate network
    # in the router because routing should be cheap and side-effect free.
    root = Path(__file__).resolve().parents[2]
    available["ibm"] = (root / ".env").exists()
    return available


def route(
    qubits: int,
    depth: int,
    accuracy: str = "exact",
    time_budget_s: float = 60.0,
    ram_gb: float = 16.0,
    vram_gb: float = 6.0,
    allow_ibm: bool = False,
    data_dir: Path | None = None,
) -> RoutingDecision:
    """Return a backend recommendation for a circuit size and accuracy mode.

    accuracy:
      - "exact": prefer noiseless CPU/GPU simulation.
      - "hardware": prefer IBM QPU semantics.
      - "estimate": choose the fastest feasible local simulator, with IBM as
        an optional recommendation when local simulation exceeds budgets.
    """
    if accuracy not in {"exact", "hardware", "estimate"}:
        raise ValueError("accuracy must be one of: exact, hardware, estimate")

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = data_dir or repo_root / "quantum_vqe" / "outputs"
    available = _availability()
    mem_gb = statevector_memory_gb(qubits)

    cpu_model = _fit_log_runtime(_read_stack_csv(data_dir / "stack_cpu.csv"))
    cudaq_model = _fit_log_runtime(_read_stack_csv(data_dir / "stack_cudaq.csv"))
    cpu_t = _predict_runtime(cpu_model, qubits, depth)
    cudaq_t = _predict_runtime(cudaq_model, qubits, depth)

    estimates: list[BackendEstimate] = []

    cpu_feasible = available["cpu"] and mem_gb <= ram_gb and (cpu_t is None or cpu_t <= time_budget_s)
    estimates.append(
        BackendEstimate(
            backend="cpu",
            available=available["cpu"],
            feasible=cpu_feasible,
            estimated_time_s=cpu_t,
            estimated_memory_gb=mem_gb,
            reason=_reason("cpu", available["cpu"], mem_gb, ram_gb, cpu_t, time_budget_s),
        )
    )

    cudaq_feasible = available["cudaq"] and mem_gb <= vram_gb and (cudaq_t is None or cudaq_t <= time_budget_s)
    estimates.append(
        BackendEstimate(
            backend="cudaq",
            available=available["cudaq"],
            feasible=cudaq_feasible,
            estimated_time_s=cudaq_t,
            estimated_memory_gb=mem_gb,
            reason=_reason("cudaq", available["cudaq"], mem_gb, vram_gb, cudaq_t, time_budget_s),
        )
    )

    ibm_feasible = available["ibm"] and allow_ibm and accuracy in {"hardware", "estimate"}
    if accuracy == "exact":
        ibm_reason = "exact/noiseless accuracy requested; IBM QPU is noisy hardware and is not eligible"
    elif ibm_feasible:
        ibm_reason = "real-QPU execution requested/allowed; result is noisy and queue-limited"
    else:
        ibm_reason = "IBM is gated; pass --allow-ibm for hardware execution recommendation"
    estimates.append(
        BackendEstimate(
            backend="ibm",
            available=available["ibm"],
            feasible=ibm_feasible,
            estimated_time_s=None,
            estimated_memory_gb=None,
            reason=ibm_reason,
        )
    )

    selected: str | None = None
    explanation = ""
    if accuracy == "hardware":
        selected = "ibm" if ibm_feasible else None
        explanation = (
            "hardware semantics requested; IBM selected"
            if selected
            else "hardware semantics requested, but IBM is not allowed or not configured"
        )
    else:
        local = [e for e in estimates if e.backend in {"cpu", "cudaq"} and e.feasible]
        if local:
            selected_estimate = min(
                local,
                key=lambda e: float("inf") if e.estimated_time_s is None else e.estimated_time_s,
            )
            selected = selected_estimate.backend
            explanation = f"selected {selected}: fastest feasible noiseless simulator within budgets"
        elif accuracy == "estimate" and ibm_feasible:
            selected = "ibm"
            explanation = "local simulation exceeds budgets; IBM selected for hardware semantics"
        else:
            selected = None
            explanation = (
                "no noiseless simulator satisfies the requested budgets"
                if accuracy == "exact"
                else "no backend satisfies the requested budgets; IBM remains gated"
            )

    return RoutingDecision(selected_backend=selected, estimates=estimates, explanation=explanation)


def _reason(
    backend: str,
    available: bool,
    mem_gb: float,
    memory_budget_gb: float,
    estimated_time_s: float | None,
    time_budget_s: float,
) -> str:
    if not available:
        return f"{backend} backend is not available in this environment"
    if mem_gb > memory_budget_gb:
        return f"estimated statevector memory {mem_gb:.2f} GB exceeds budget {memory_budget_gb:.2f} GB"
    if estimated_time_s is not None and estimated_time_s > time_budget_s:
        return f"estimated runtime {estimated_time_s:.2f}s exceeds budget {time_budget_s:.2f}s"
    return "within configured time and memory budgets"
