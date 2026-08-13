"""Environment self-check and local capability profiler.

Run this after cloning the repository to discover what the current machine can
execute: CPU simulation, CUDA-Q/GPU simulation, and IBM Runtime configuration.
The script performs lightweight smoke benchmarks and writes a machine-readable
profile for the resource-aware router.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_dev.router import statevector_memory_gb
from run_ibm_cloud import load_token

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _module_status(name: str) -> dict[str, object]:
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:  # noqa: BLE001 - self-check should report all failures.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _clean_output(text: str, max_chars: int = 4000) -> str:
    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "\n...[truncated]"


def _run_command(cmd: list[str], timeout: int = 5) -> dict[str, object]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _clean_output(proc.stdout),
            "stderr": _clean_output(proc.stderr),
        }
    except FileNotFoundError:
        return {"available": False, "error": "command not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "command timed out"}


def _memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return None


def _cpu_smoke(max_qubits: int, depth: int) -> dict[str, object]:
    try:
        import numpy as np
        from quantum_stack_benchmark import run_cpu

        rows = []
        for n in range(4, max_qubits + 1, 4):
            t0 = time.perf_counter()
            runtime, exp = run_cpu(n, depth)
            rows.append(
                {
                    "qubits": n,
                    "depth": depth,
                    "runtime_s": runtime,
                    "wall_s": time.perf_counter() - t0,
                    "expectation": float(exp),
                    "statevector_gb": statevector_memory_gb(n),
                }
            )
        recommended = max((r["qubits"] for r in rows if r["runtime_s"] <= 1.0), default=None)
        return {"available": True, "rows": rows, "recommended_under_1s": recommended}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _cudaq_smoke(max_qubits: int, depth: int) -> dict[str, object]:
    try:
        import cudaq
        from quantum_stack_benchmark import run_cudaq

        cudaq.set_target("nvidia")
        target = cudaq.get_target()
        rows = []
        for n in range(4, max_qubits + 1, 4):
            runtime, exp = run_cudaq(n, depth)
            rows.append(
                {
                    "qubits": n,
                    "depth": depth,
                    "runtime_s": runtime,
                    "expectation": float(exp),
                    "statevector_gb": statevector_memory_gb(n),
                }
            )
        recommended = max((r["qubits"] for r in rows if r["runtime_s"] <= 1.0), default=None)
        return {
            "available": True,
            "target": target.name,
            "num_qpus": target.num_qpus(),
            "rows": rows,
            "recommended_under_1s": recommended,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _ibm_status(check_network: bool) -> dict[str, object]:
    token = load_token()
    status: dict[str, object] = {"configured": bool(token)}
    if not token or not check_network:
        return status
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backends = []
        for backend in sorted(service.backends(), key=lambda b: b.name):
            st = service.backend(backend.name).status()
            backends.append(
                {
                    "name": backend.name,
                    "qubits": backend.num_qubits,
                    "pending_jobs": st.pending_jobs,
                    "operational": st.operational,
                }
            )
        status.update({"network_ok": True, "backends": backends})
    except Exception as exc:  # noqa: BLE001
        status.update({"network_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return status


def _recommend(profile: dict[str, object]) -> list[str]:
    recommendations = []
    modules = profile["python_modules"]
    cpu_ok = bool(modules["qiskit"]["available"])
    cudaq_ok = bool(profile["cudaq_smoke"]["available"])
    ibm_cfg = bool(profile["ibm"]["configured"])

    if cpu_ok:
        n_cpu = profile["cpu_smoke"].get("recommended_under_1s")
        recommendations.append(f"CPU simulation available; smoke test suggests <=1s up to n={n_cpu}.")
    else:
        recommendations.append("CPU simulation stack is incomplete; install requirements.txt.")

    if cudaq_ok:
        n_gpu = profile["cudaq_smoke"].get("recommended_under_1s")
        recommendations.append(f"CUDA-Q GPU path available; smoke test suggests <=1s up to n={n_gpu}.")
    else:
        recommendations.append("CUDA-Q GPU path unavailable here; use WSL2/Linux/Docker with CUDA-Q for GPU runs.")

    if ibm_cfg:
        recommendations.append("IBM token is configured; QPU execution can be requested explicitly.")
    else:
        recommendations.append("IBM token is not configured; set QISKIT_IBM_TOKEN for cloud QPU workflows.")

    return recommendations


def _write_markdown(profile: dict[str, object], path: Path) -> None:
    lines = [
        "# Quantum Environment Profile",
        "",
        f"- Python: `{profile['python']['version']}`",
        f"- Platform: `{profile['system']['platform']}`",
        f"- RAM: `{profile['system']['ram_gb']}` GB",
        "",
        "## Module Availability",
        "",
        "| Module | Available | Version / Error |",
        "|---|---:|---|",
    ]
    for name, status in profile["python_modules"].items():
        detail = status.get("version") if status.get("available") else status.get("error")
        lines.append(f"| `{name}` | {status.get('available')} | `{detail}` |")

    lines += ["", "## Hardware", ""]
    nvidia = profile["hardware"]["nvidia_smi"]
    gpu_query = profile["hardware"]["nvidia_gpu_query"]
    lines.append(f"- `nvidia-smi`: {nvidia.get('available')}")
    if gpu_query.get("stdout"):
        lines.append(f"- NVIDIA GPU: `{gpu_query['stdout']}`")
    lines.append(f"- WSL: {profile['hardware']['wsl'].get('available')}")

    lines += ["", "## Smoke Benchmarks", ""]
    for key, title in [("cpu_smoke", "CPU"), ("cudaq_smoke", "CUDA-Q GPU")]:
        smoke = profile[key]
        lines.append(f"### {title}")
        if not smoke.get("available"):
            lines.append(f"- unavailable: `{smoke.get('error')}`")
            lines.append("")
            continue
        lines += ["", "| Qubits | Depth | Runtime (s) | <Z0> | Statevector GB |", "|---:|---:|---:|---:|---:|"]
        for row in smoke["rows"]:
            lines.append(
                f"| {row['qubits']} | {row['depth']} | {row['runtime_s']:.6f} | "
                f"{row['expectation']:.6f} | {row['statevector_gb']:.6f} |"
            )
        lines.append("")

    lines += ["## Recommendations", ""]
    for rec in profile["recommendations"]:
        lines.append(f"- {rec}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check the local quantum development environment.")
    parser.add_argument("--max-qubits", type=int, default=12, help="Largest n for smoke benchmarks.")
    parser.add_argument("--depth", type=int, default=1, help="Circuit depth for smoke benchmarks.")
    parser.add_argument("--check-ibm", action="store_true", help="Query IBM backends if a token is configured.")
    parser.add_argument("--json", default="env_profile.json", help="Output JSON filename under quantum_vqe/outputs.")
    parser.add_argument("--markdown", default="env_profile.md", help="Output Markdown filename under quantum_vqe/outputs.")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    profile: dict[str, object] = {
        "python": {"version": sys.version.replace("\n", " ")},
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "ram_gb": None if _memory_gb() is None else round(float(_memory_gb()), 3),
        },
        "python_modules": {
            "numpy": _module_status("numpy"),
            "scipy": _module_status("scipy"),
            "qiskit": _module_status("qiskit"),
            "qiskit_aer": _module_status("qiskit_aer"),
            "pennylane": _module_status("pennylane"),
            "cirq": _module_status("cirq"),
            "cudaq": _module_status("cudaq"),
            "cupy": _module_status("cupy"),
        },
        "hardware": {
            "nvidia_smi": _run_command(["nvidia-smi"], timeout=8),
            "nvidia_gpu_query": _run_command(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                timeout=8,
            ),
            "wsl": _run_command(["wsl", "-l", "-v"], timeout=8),
        },
        "cpu_smoke": _cpu_smoke(args.max_qubits, args.depth),
        "cudaq_smoke": _cudaq_smoke(args.max_qubits, args.depth),
        "ibm": _ibm_status(args.check_ibm),
    }
    profile["recommendations"] = _recommend(profile)

    json_path = OUT_DIR / args.json
    md_path = OUT_DIR / args.markdown
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _write_markdown(profile, md_path)

    print("Quantum environment self-check")
    print("=" * 40)
    for rec in profile["recommendations"]:
        print(f"- {rec}")
    print(f"\nprofile_json={json_path}")
    print(f"profile_markdown={md_path}")


if __name__ == "__main__":
    main()
