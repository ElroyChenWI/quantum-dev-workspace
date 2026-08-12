"""
quantum_stack_benchmark — 量子平台堆疊四層效能 benchmark
=========================================================

用**同一個 parameterized hardware-efficient ansatz（HEA）**，在四層後端上
量測「期望值計算」的時間，展示量子平台堆疊的差異：

    L1 CPU simulator            → 正確性基準（Qiskit StatevectorEstimator）
    L2 naive GPU                → 一般 GPU 平行化的限制（手寫 cupy statevector）
    L3 CUDA-Q / cuStateVec      → 專用量子模擬 kernel 的優勢（nvidia GPU target）
    L4 IBM Quantum              → 真實 QPU 執行（shallow，省額度）

電路（每一層）：
    RY(θ) 在每個 qubit → RZ(θ) 在每個 qubit → CNOT 鏈（i, i+1）
    重複 depth 次
量測：⟨Z₀⟩ 期望值

執行方式：
    # L1（Windows，有 qiskit）
    python quantum_vqe/quantum_stack_benchmark.py --cpu

    # L2/L3（WSL 的 cudaq 環境，有 cupy + cudaq）
    python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq

    # L4（Windows，需 .env 的 IBM token；建議小 n、淺 depth 省額度）
    python quantum_vqe/quantum_stack_benchmark.py --ibm

    # 全部（會自動偵測可用層）
    python quantum_vqe/quantum_stack_benchmark.py --all

輸出：
    outputs/stack_<layer>.csv   # 每層的 (n, time) 資料
    outputs/stack_benchmark.png # 多層比較圖（log 尺規）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"
ROOT = Path(__file__).resolve().parents[1]


def build_hea_qiskit(n: int, depth: int, params: np.ndarray):
    """HEA：RY+RZ 全 qubit → CNOT 鏈，重複 depth 次（Qiskit 版本，固定參數）。"""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    idx = 0
    for _ in range(depth):
        for i in range(n):
            qc.ry(params[idx], i)
            idx += 1
        for i in range(n):
            qc.rz(params[idx], i)
            idx += 1
        for i in range(n - 1):
            qc.cx(i, i + 1)
    return qc


def z0_observable_qiskit(n: int):
    from qiskit.quantum_info import SparsePauliOp

    label = "Z" + "I" * (n - 1)
    return SparsePauliOp.from_list([(label, 1.0)])


# ---------------------------------------------------------------------------
# L1: CPU simulator（Qiskit StatevectorEstimator）
# ---------------------------------------------------------------------------
def benchmark_cpu(ns: list[int], depth: int, params_factory) -> list[float]:
    from qiskit.primitives import StatevectorEstimator

    est = StatevectorEstimator()
    times = []
    for n in ns:
        params = params_factory(n, depth)
        qc = build_hea_qiskit(n, depth, params)
        obs = z0_observable_qiskit(n)
        # warm up
        est.run([(qc, obs)]).result()
        t0 = time.perf_counter()
        est.run([(qc, obs)]).result()
        times.append(time.perf_counter() - t0)
        print(f"  [CPU ] n={n:2d} depth={depth} → {times[-1]:8.4f} s")
    return times


# ---------------------------------------------------------------------------
# L2: naive GPU simulator（手寫 cupy statevector，無優化）
# ---------------------------------------------------------------------------
def _naive_apply_single(sv, q, g, n):
    """單 qubit gate（2x2），用 gather/scatter 實作——刻意不優化。"""
    import cupy as cp

    stride = 1 << q
    idx = cp.arange(sv.size, dtype=cp.int64)
    lo = idx[(idx & stride) == 0]
    hi = lo | stride
    v_lo = sv[lo]
    v_hi = sv[hi]
    out = sv.copy()
    out[lo] = g[0, 0] * v_lo + g[0, 1] * v_hi
    out[hi] = g[1, 0] * v_lo + g[1, 1] * v_hi
    return out


def _naive_apply_cnot(sv, c, t, n):
    """CNOT：control=1 時對 target 做 swap（gather/scatter）。"""
    import cupy as cp

    stride_c, stride_t = 1 << c, 1 << t
    idx = cp.arange(sv.size, dtype=cp.int64)
    ctrl_sel = (idx & stride_c) != 0
    lo = idx[ctrl_sel & ((idx & stride_t) == 0)]
    hi = lo ^ stride_t
    out = sv.copy()
    tmp = out[lo].copy()
    out[lo] = out[hi]
    out[hi] = tmp
    return out


def benchmark_naive_gpu(ns: list[int], depth: int, params_factory) -> list[float]:
    import cupy as cp

    times = []
    for n in ns:
        dim = 1 << n
        params = params_factory(n, depth)
        sv = cp.zeros(dim, dtype=cp.complex128)
        sv[0] = 1.0
        idx = 0
        ry = lambda t: cp.asarray(
            np.array([[np.cos(t / 2), -np.sin(t / 2)],
                      [np.sin(t / 2), np.cos(t / 2)]], dtype=np.complex128)
        )
        rz = lambda t: cp.asarray(
            np.array([[np.exp(-1j * t / 2), 0],
                      [0, np.exp(1j * t / 2)]], dtype=np.complex128)
        )

        def evolve():
            s = sv
            k = 0
            for _ in range(depth):
                for q in range(n):
                    s = _naive_apply_single(s, q, ry(params[k]), n)
                    k += 1
                for q in range(n):
                    s = _naive_apply_single(s, q, rz(params[k]), n)
                    k += 1
                for q in range(n - 1):
                    s = _naive_apply_cnot(s, q, q + 1, n)
            return s

        # warm up
        evolve()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        s = evolve()
        cp.cuda.Stream.null.synchronize()
        # ⟨Z₀⟩
        i0 = cp.arange(dim)
        sign = cp.where((i0 & 1) == 0, 1.0, -1.0)
        exp_z0 = float(cp.sum(sign * cp.abs(s) ** 2))
        times.append(time.perf_counter() - t0)
        print(f"  [GPU ] n={n:2d} depth={depth} → {times[-1]:8.4f} s  <Z0>={exp_z0:.4f}")
    return times


# ---------------------------------------------------------------------------
# L3: CUDA-Q / cuStateVec（nvidia GPU target）
# ---------------------------------------------------------------------------
def benchmark_cudaq(ns: list[int], depth: int, params_factory) -> list[float]:
    import cudaq

    cudaq.set_target("nvidia")
    from cudaq import spin

    times = []
    for n in ns:
        params = params_factory(n, depth)

        @cudaq.kernel
        def hea(p: list[float]):
            q = cudaq.qvector(n)
            k = 0
            for _ in range(depth):
                for i in range(n):
                    ry(p[k], q[i])
                    k += 1
                for i in range(n):
                    rz(p[k], q[i])
                    k += 1
                for i in range(n - 1):
                    x.ctrl(q[i], q[i + 1])

        # n-qubit observable ⟨Z₀⟩
        H = spin.z(0)
        for i in range(1, n):
            H = H * spin.i(i)
        exp_val = cudaq.observe(hea, H, params).expectation()  # warm up
        t0 = time.perf_counter()
        exp_val = cudaq.observe(hea, H, params).expectation()
        times.append(time.perf_counter() - t0)
        print(f"  [CUDAQ] n={n:2d} depth={depth} → {times[-1]:8.4f} s  <Z0>={exp_val:.4f}")
    return times


# ---------------------------------------------------------------------------
# L4: IBM Quantum（真實 QPU，shallow，省額度）
# ---------------------------------------------------------------------------
def benchmark_ibm(ns: list[int], depth: int, params_factory) -> list[float]:
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from run_ibm_cloud import load_token
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService

    token = load_token()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = [b for b in service.backends() if service.backend(b.name).status().operational]
    if not backends:
        raise RuntimeError("沒有可用的 IBM backend")
    backend = min(backends, key=lambda b: service.backend(b.name).status().pending_jobs)
    print(f"  使用 IBM backend: {backend.name}")

    times = []
    for n in ns:
        params = params_factory(n, depth)
        qc = build_hea_qiskit(n, depth, params)
        obs = z0_observable_qiskit(n)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_qc = pm.run(qc)
        isa_obs = obs.apply_layout(isa_qc.layout)
        estimator = Estimator(backend)
        job = estimator.run([(isa_qc, isa_obs)])
        print(f"  [IBM ] n={n:2d} depth={depth} → job {job.job_id()}（排隊中）")
        t0 = time.perf_counter()
        result = job.result()
        times.append(time.perf_counter() - t0)
        val = float(result[0].data.evs[0])
        print(f"          → {times[-1]:8.4f} s（含排隊） <Z0>={val:.4f}")
    return times


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def make_params_factory(rng_seed: int = 42):
    def factory(n: int, depth: int):
        rng = np.random.default_rng(rng_seed + n)
        return rng.uniform(-0.5, 0.5, n * depth * 2)

    return factory


def available_backends() -> dict:
    bk = {"cpu": False, "naive_gpu": False, "cudaq": False, "ibm": False}
    try:
        import qiskit

        bk["cpu"] = True
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401

        bk["naive_gpu"] = True
    except ImportError:
        pass
    try:
        import cudaq  # noqa: F401

        bk["cudaq"] = True
    except ImportError:
        pass
    if (ROOT / ".env").exists():
        bk["ibm"] = True
    return bk


def main():
    parser = argparse.ArgumentParser(description="量子平台堆疊四層 benchmark")
    parser.add_argument("--min_qubits", type=int, default=4)
    parser.add_argument("--max_qubits", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cpu", action="store_true", help="跑 L1 CPU")
    parser.add_argument("--naive-gpu", action="store_true", help="跑 L2 naive GPU")
    parser.add_argument("--cudaq", action="store_true", help="跑 L3 CUDA-Q")
    parser.add_argument("--ibm", action="store_true", help="跑 L4 IBM（省額度）")
    parser.add_argument("--all", action="store_true", help="跑所有可用層")
    parser.add_argument("--plot", action="store_true", help="只重畫比較圖（讀取既有 CSV）")
    args = parser.parse_args()

    ns = list(range(args.min_qubits, args.max_qubits + 1, 4))

    if args.plot:
        _plot(ns)
        return

    params_factory = make_params_factory()
    bk = available_backends()

    if args.all:
        args.cpu = bk["cpu"]
        args.naive_gpu = bk["naive_gpu"]
        args.cudaq = bk["cudaq"]
        args.ibm = bk["ibm"]

    print("=" * 64)
    print("quantum_stack_benchmark — 四層後端期望值計算時間")
    print("=" * 64)
    print(f"qubits: {ns}, depth: {args.depth}")
    print(f"偵測到可用層: {[k for k, v in bk.items() if v]}")

    results = {}

    if args.cpu:
        print("\n--- L1: CPU simulator (Qiskit) ---")
        results["cpu"] = benchmark_cpu(ns, args.depth, params_factory)
    if args.naive_gpu:
        print("\n--- L2: naive GPU (cupy) ---")
        results["naive_gpu"] = benchmark_naive_gpu(ns, args.depth, params_factory)
    if args.cudaq:
        print("\n--- L3: CUDA-Q / cuStateVec (nvidia) ---")
        results["cudaq"] = benchmark_cudaq(ns, args.depth, params_factory)
    if args.ibm:
        print("\n--- L4: IBM Quantum QPU ---")
        ibm_ns = [args.min_qubits]  # 省額度：只跑最小 n
        results["ibm"] = benchmark_ibm(ibm_ns, min(args.depth, 1), params_factory)

    if not results:
        print("沒有執行任何層。用 --all 或指定 --cpu/--naive-gpu/--cudaq/--ibm。")
        return

    # 儲存 CSV
    OUT_DIR.mkdir(exist_ok=True)
    for layer, times in results.items():
        np.savetxt(
            OUT_DIR / f"stack_{layer}.csv",
            np.column_stack([ns[: len(times)], times]),
            header="qubits,time_s",
            comments="",
        )

    # 畫圖（合併所有已存在層的 CSV，跨環境）
    _plot(ns)


def _plot(ns: list[int]):
    import matplotlib

    # 跨 Windows / WSL 通用：用 ASCII 標籤避免缺中文字型
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    layers = {
        "cpu": ("CPU simulator", "#1f77b4", "o-"),
        "naive_gpu": ("naive GPU", "#d62728", "s-"),
        "cudaq": ("CUDA-Q / cuStateVec", "#2ca02c", "^-"),
        "ibm": ("IBM QPU", "#9467bd", "d-"),
    }

    found = [k for k in layers if (OUT_DIR / f"stack_{k}.csv").exists()]
    if not found:
        print("沒有可畫的資料（無 stack_*.csv）")
        return

    plt.figure(figsize=(9, 5.5))
    for key in found:
        label, color, marker = layers[key]
        data = np.atleast_2d(np.loadtxt(OUT_DIR / f"stack_{key}.csv", skiprows=1))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        plt.plot(data[:, 0], data[:, 1], marker, color=color, label=label)

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1e}"))
    plt.xlabel("Qubits N")
    plt.ylabel("Expectation time (s, log)")
    plt.title("Quantum Stack Benchmark - same HEA circuit, 4 backends")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    png = OUT_DIR / "stack_benchmark.png"
    plt.savefig(png, dpi=130)
    plt.close()
    print(f"\n[chart] saved: {png}")


if __name__ == "__main__":
    main()
