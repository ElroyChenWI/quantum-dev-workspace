"""IBM Quantum cloud account and backend capability profiler.

This script keeps remote-QPU state separate from local environment state. It
checks whether IBM Runtime is configured, records accessible backend status, and
captures account usage when the installed Runtime client exposes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ibm_cloud import load_token

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _usage(service: Any) -> dict[str, Any]:
    try:
        raw = _jsonable(service.usage())
        if isinstance(raw, dict):
            raw = {
                key: value
                for key, value in raw.items()
                if key not in {"instance_id", "plan_id"} and not key.lower().endswith("_id")
            }
        return {"available": True, "data": raw}
    except Exception as exc:  # noqa: BLE001 - account APIs vary by plan/client version.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _backend_rows(service: Any, backend_filter: str | None) -> list[dict[str, Any]]:
    rows = []
    backends = service.backends()
    for backend in sorted(backends, key=lambda b: b.name):
        if backend_filter and backend.name != backend_filter:
            continue
        status = service.backend(backend.name).status()
        rows.append(
            {
                "name": backend.name,
                "qubits": backend.num_qubits,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "status_msg": getattr(status, "status_msg", ""),
            }
        )
    return rows


def build_profile(backend_filter: str | None = None, skip_backends: bool = False) -> dict[str, Any]:
    token = load_token()
    profile: dict[str, Any] = {
        "ibm": {
            "configured": bool(token),
            "network_ok": None,
            "active_instance": None,
            "usage": {"available": False, "error": "not queried"},
            "backends": [],
        }
    }
    if not token:
        profile["ibm"]["network_ok"] = False
        profile["ibm"]["usage"] = {"available": False, "error": "IBM token is not configured"}
        return profile

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        profile["ibm"]["network_ok"] = True
        try:
            active_instance = service.active_instance()
            profile["ibm"]["active_instance"] = "available" if active_instance else None
        except Exception as exc:  # noqa: BLE001
            profile["ibm"]["active_instance"] = f"unavailable: {type(exc).__name__}: {exc}"
        profile["ibm"]["usage"] = _usage(service)
        if not skip_backends:
            profile["ibm"]["backends"] = _backend_rows(service, backend_filter)
    except Exception as exc:  # noqa: BLE001
        profile["ibm"]["network_ok"] = False
        profile["ibm"]["error"] = f"{type(exc).__name__}: {exc}"
    return profile


def _write_markdown(profile: dict[str, Any], path: Path) -> None:
    ibm = profile["ibm"]
    lines = [
        "# Quantum Cloud Profile",
        "",
        "## IBM Quantum",
        "",
        f"- Configured: `{ibm.get('configured')}`",
        f"- Network OK: `{ibm.get('network_ok')}`",
        f"- Active instance: `{ibm.get('active_instance')}`",
        "",
        "## Usage",
        "",
    ]
    usage = ibm.get("usage", {})
    if usage.get("available"):
        lines.append("```json")
        lines.append(json.dumps(usage.get("data"), indent=2))
        lines.append("```")
    else:
        lines.append(f"- unavailable: `{usage.get('error')}`")

    lines += [
        "",
        "## Backends",
        "",
        "| Backend | Qubits | Operational | Pending Jobs | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for backend in ibm.get("backends", []):
        lines.append(
            f"| `{backend['name']}` | {backend['qubits']} | {backend['operational']} | "
            f"{backend['pending_jobs']} | `{backend.get('status_msg', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile IBM Quantum account and backend availability.")
    parser.add_argument("--backend", help="Only record one backend, e.g. ibm_kingston.")
    parser.add_argument("--skip-backends", action="store_true", help="Only check account/usage, not backend status.")
    parser.add_argument("--json", default="cloud_profile.json", help="Output JSON filename under quantum_vqe/outputs.")
    parser.add_argument("--markdown", default="cloud_profile.md", help="Output Markdown filename under quantum_vqe/outputs.")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    profile = build_profile(backend_filter=args.backend, skip_backends=args.skip_backends)
    json_path = OUT_DIR / args.json
    md_path = OUT_DIR / args.markdown
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _write_markdown(profile, md_path)

    ibm = profile["ibm"]
    print("Quantum cloud self-check")
    print("=" * 40)
    print(f"ibm_configured={ibm.get('configured')}")
    print(f"ibm_network_ok={ibm.get('network_ok')}")
    print(f"backend_count={len(ibm.get('backends', []))}")
    print(f"usage_available={ibm.get('usage', {}).get('available')}")
    print(f"\nprofile_json={json_path}")
    print(f"profile_markdown={md_path}")


if __name__ == "__main__":
    main()
