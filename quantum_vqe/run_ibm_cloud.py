"""
IBM Quantum 雲端連線
====================

連接 IBM Quantum 平台，列出可用 backend，並在雲端跑 H2 的能量，
與本機模擬結果對照——展示「跨過指數牆」的最後一步。

執行方式：
    python quantum_vqe/run_ibm_cloud.py --list        # 只列出可用 backend
    python quantum_vqe/run_ibm_cloud.py               # 連線 + 跑 H2 能量
    python quantum_vqe/run_ibm_cloud.py --backend ibm_brisbane   # 指定 backend

Token 從專案根目錄的 .env（QISKIT_IBM_TOKEN）讀取。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2_hamiltonian import INITIAL_PARAMS, PAULI_TERMS, exact_ground_state_energy


def load_token() -> str:
    """從 .env 讀取 token（避免硬編碼在程式碼裡）。"""
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "QISKIT_IBM_TOKEN" and v.strip():
                    return v.strip()
    return os.environ.get("QISKIT_IBM_TOKEN", "")


def build_h2_ansatz():
    """H2 的 4 參數 ansatz（與本機 VQE 相同）。"""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("t", 4)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    return qc


def build_h2_observable():
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_list([(p, c) for p, c in PAULI_TERMS])


def main():
    parser = argparse.ArgumentParser(description="IBM Quantum 雲端連線")
    parser.add_argument("--list", action="store_true", help="只列出可用 backend")
    parser.add_argument("--backend", type=str, default=None, help="指定 backend 名稱")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("找不到 token。請在專案根目錄 .env 設定 QISKIT_IBM_TOKEN")
        sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService

    print("=" * 60)
    print("連接 IBM Quantum ...")
    print("=" * 60)
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    backends = sorted(service.backends(), key=lambda b: b.name)
    print(f"\n可用 backend 共 {len(backends)} 個：")
    for b in backends:
        status = service.backend(b.name).status()
        print(f"  {b.name:24s} qubits={b.num_qubits:5d}  "
              f"pending_jobs={status.pending_jobs:4d}  "
              f"operational={status.operational}")

    if args.list:
        return

    # 選 backend：指定 > 預設（優先選可用的）
    if args.backend:
        backend_name = args.backend
    else:
        # 優先挑 operational 且排隊最短的
        candidates = [b for b in backends if service.backend(b.name).status().operational]
        if not candidates:
            print("沒有可用的 backend")
            sys.exit(1)
        backend_name = min(
            candidates,
            key=lambda b: service.backend(b.name).status().pending_jobs,
        ).name

    backend = service.backend(backend_name)
    print(f"\n使用 backend: {backend_name}（{backend.num_qubits} qubits）")

    # 準備 H2 電路
    ansatz = build_h2_ansatz()
    observable = build_h2_observable()
    exact = exact_ground_state_energy()
    print(f"H2 精確基態能量（對角化）: {exact:.6f} Ha")
    print(f"在參數 {INITIAL_PARAMS} 下測量能量期望值...")

    # 本機參考值
    from qiskit.primitives import StatevectorEstimator

    local = StatevectorEstimator().run(
        [(ansatz, observable, [INITIAL_PARAMS])]
    ).result()[0].data.evs[0]
    print(f"本機 StatevectorEstimator: {local:.6f} Ha")

    # 雲端 EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    # IBM 要求電路先轉譯成目標硬體的 ISA（原生閘）才能執行
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(ansatz)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    estimator = Estimator(backend)
    print("提交到 IBM 雲端（可能需排隊）...")
    job = estimator.run([(isa_circuit, isa_observable, [INITIAL_PARAMS])])
    print(f"  job id: {job.job_id()}")
    result = job.result()
    cloud = float(result[0].data.evs[0])
    print(f"IBM 雲端 Estimator      : {cloud:.6f} Ha")
    print(f"誤差（vs 本機）         : {abs(cloud - local):.2e} Ha")

    print(f"\n結論：本機與 IBM 真實量子處理器的結果接近但不完全相同。")
    print(f"差異 {abs(cloud - local):.3f} Ha 正是『硬體雜訊』的影響：")
    print("模擬器是無雜訊的理想值，真硬體必然有誤差——")
    print("這正是需要研究真實量子硬體的原因。")
    print("同一段量子程式可以從本機直接搬到雲端/真硬體執行（跨過指數牆）。")


if __name__ == "__main__":
    main()
