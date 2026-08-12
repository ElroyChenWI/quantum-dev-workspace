"""
quantum_stack_benchmark — 量子平台堆疊四層效能 benchmark（技術審查版）
=======================================================================

用**同一個 parameterized hardware-efficient ansatz（HEA）**，在四層後端上
量測「期望值計算」的時間，展示量子平台堆疊的差異：

    L1 CPU simulator            → 正確性基準（Qiskit StatevectorEstimator）
    L2 naive GPU                → 一般 GPU 平行化的限制（手寫 cupy，educational baseline）
    L3 CUDA-Q / cuStateVec      → 專用量子模擬 kernel 的優勢（nvidia GPU target）
    L4 IBM Quantum              → 真實 QPU 執行（語義比較：noisy hardware result，不比秒數）

測量方法（專業級）：
    - 每個 backend 先印出 device / target（證明真的用對硬體）
    - GPU 計時強制 synchronization（避免 kernel async 造成時間偏低）
    - warmup 後取多次 median（避免 compile/init/allocate 影響）
    - 掃描多個 depth 與 qubit 數
    - 輸出統一格式：backend, device, target, precision, qubits, depth, runtime_s, expectation

執行方式：
    python quantum_vqe/quantum_stack_benchmark.py --cpu --verify-target
    python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq --verify-target
    python quantum_vqe/quantum_stack_benchmark.py --all --verify-target --depths 1,3,6,10 --max_qubits 24
    python quantum_vqe/quantum_stack_benchmark.py --ibm            # 語義比較，省額度
    python quantum_vqe/quantum_stack_benchmark.py --plot           # 只重畫圖

輸出：
    outputs/stack_results.csv    # 統一格式結果
    outputs/stack_benchmark.png  # 依 depth 分 subplot 的比較圖（不含 IBM runtime）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"
ROOT = Path(__file__).resolve().parents[1]

# 統一 CSV 欄位
CSV_HEADER = "backend,device,target,precision,qubits,depth,runtime_s,expectation"


# ---------------------------------------------------------------------------
# 共用電路
# ---------------------------------------------------------------------------
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

    # Qiskit Pauli 字串最右邊是 qubit 0；naive GPU 用 (i & 1)、CUDA-Q 用 spin.z(0) 都是量 qubit 0。
    # 必須用 "I"*(n-1)+"Z"，否則會量到最高位 qubit，三層就不是同一個 <Z0>。
    return SparsePauliOp.from_list([("I" * (n - 1) + "Z", 1.0)])


def make_params(n: int, depth: int, seed: int = 42):
    rng = np.random.default_rng(seed + n * 1000 + depth)
    return rng.uniform(-0.5, 0.5, n * depth * 2)


def timed(fn, warmup: int, repeats: int) -> float:
    """warmup 後取中位數。回傳秒數。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# L1: CPU simulator（Qiskit StatevectorEstimator）
# ---------------------------------------------------------------------------
def backend_cpu() -> tuple[str, str, str]:
    import qiskit

    return "qiskit", "cpu", f"StatevectorEstimator qiskit-{qiskit.__version__}"


def run_cpu(n: int, depth: int):
    from qiskit.primitives import StatevectorEstimator

    params = make_params(n, depth)
    qc = build_hea_qiskit(n, depth, params)
    obs = z0_observable_qiskit(n)
    est = StatevectorEstimator()

    def fn():
        res = est.run([(qc, obs)]).result()
        return float(np.atleast_1d(np.asarray(res[0].data.evs))[0])

    return timed(fn, WARMUP, REPEATS), fn()


# ---------------------------------------------------------------------------
# L2: naive GPU simulator（手寫 cupy，educational baseline）
# ---------------------------------------------------------------------------
def backend_naive_gpu() -> tuple[str, str, str]:
    import cupy as cp

    dev = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(dev)
    return "naive_gpu", f"gpu:{dev}", props["name"].decode()


def _naive_apply_single(sv, q, g, n):
    import cupy as cp

    stride = 1 << q
    idx = cp.arange(sv.size, dtype=cp.int64)
    lo = idx[(idx & stride) == 0]
    hi = lo | stride
    v_lo, v_hi = sv[lo], sv[hi]
    out = sv.copy()
    out[lo] = g[0, 0] * v_lo + g[0, 1] * v_hi
    out[hi] = g[1, 0] * v_lo + g[1, 1] * v_hi
    return out


def _naive_apply_cnot(sv, c, t, n):
    import cupy as cp

    stride_c, stride_t = 1 << c, 1 << t
    idx = cp.arange(sv.size, dtype=cp.int64)
    ctrl = (idx & stride_c) != 0
    lo = idx[ctrl & ((idx & stride_t) == 0)]
    hi = lo ^ stride_t
    out = sv.copy()
    tmp = out[lo].copy()
    out[lo] = out[hi]
    out[hi] = tmp
    return out


def run_naive_gpu(n: int, depth: int):
    import cupy as cp

    params = make_params(n, depth)
    dim = 1 << n
    sv = cp.zeros(dim, dtype=cp.complex128)
    sv[0] = 1.0
    ry = lambda t: cp.asarray(
        np.array([[np.cos(t / 2), -np.sin(t / 2)],
                  [np.sin(t / 2), np.cos(t / 2)]], dtype=np.complex128)
    )
    rz = lambda t: cp.asarray(
        np.array([[np.exp(-1j * t / 2), 0],
                  [0, np.exp(1j * t / 2)]], dtype=np.complex128)
    )

    def fn():
        k = 0
        s = sv
        for _ in range(depth):
            for q in range(n):
                s = _naive_apply_single(s, q, ry(params[k]), n)
                k += 1
            for q in range(n):
                s = _naive_apply_single(s, q, rz(params[k]), n)
                k += 1
            for q in range(n - 1):
                s = _naive_apply_cnot(s, q, q + 1, n)
        # 強制 sync：確保 GPU kernel 全部完成才計時結束
        cp.cuda.Stream.null.synchronize()
        i0 = cp.arange(dim)
        sign = cp.where((i0 & 1) == 0, 1.0, -1.0)
        exp = float(cp.sum(sign * cp.abs(s) ** 2))
        cp.cuda.Stream.null.synchronize()
        return exp

    return timed(fn, WARMUP, REPEATS), fn()


# ---------------------------------------------------------------------------
# L3: CUDA-Q / cuStateVec（nvidia GPU target）
# ---------------------------------------------------------------------------
def backend_cudaq() -> tuple[str, str, str]:
    import cudaq

    cudaq.set_target("nvidia")
    target = cudaq.get_target().name
    return "cudaq", "gpu", f"target={target} ({cudaq.get_target().num_qpus()} QPU)"


def run_cudaq(n: int, depth: int):
    import cudaq
    from cudaq import spin

    cudaq.set_target("nvidia")
    params = make_params(n, depth)

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

    H = spin.z(0)
    for i in range(1, n):
        H = H * spin.i(i)

    def fn():
        # .expectation() 會 materialize 結果 = 強制同步
        return cudaq.observe(hea, H, params).expectation()

    return timed(fn, WARMUP, REPEATS), float(fn())


# ---------------------------------------------------------------------------
# L4: IBM Quantum（真實 QPU — 語義比較，不比秒數）
# ---------------------------------------------------------------------------
def backend_ibm() -> tuple[str, str, str]:
    return "ibm", "qpu", "QiskitRuntimeService (ibm_quantum_platform)"


def run_ibm(n: int, depth: int):
    sys.path.insert(0, str(ROOT))
    from run_ibm_cloud import load_token
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService

    token = load_token()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = [b for b in service.backends() if service.backend(b.name).status().operational]
    if not backends:
        raise RuntimeError("無可用 IBM backend")
    backend = min(backends, key=lambda b: service.backend(b.name).status().pending_jobs)

    params = make_params(n, depth)
    qc = build_hea_qiskit(n, depth, params)
    obs = z0_observable_qiskit(n)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    isa_obs = obs.apply_layout(isa_qc.layout)
    estimator = Estimator(backend)
    job = estimator.run([(isa_qc, isa_obs)])
    result = job.result()  # 阻塞直到完成（含 queue time）
    ev = float(np.atleast_1d(np.asarray(result[0].data.evs))[0])
    return float(np.nan), ev  # runtime 標 NaN：不比秒數


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
WARMUP = 1
REPEATS = 3


def available_backends() -> dict:
    bk = {"cpu": False, "naive_gpu": False, "cudaq": False, "ibm": False}
    try:
        import qiskit  # noqa: F401

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
    global WARMUP, REPEATS

    parser = argparse.ArgumentParser(description="量子平台堆疊四層 benchmark（技術審查版）")
    parser.add_argument("--min_qubits", type=int, default=4)
    parser.add_argument("--max_qubits", type=int, default=24)
    parser.add_argument("--depths", type=str, default="3", help="逗號分隔的 depth 清單，如 1,3,6,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--verify-target", action="store_true", help="印出每個 backend 的 device/target 證明")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--naive-gpu", action="store_true")
    parser.add_argument("--cudaq", action="store_true")
    parser.add_argument("--ibm", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--plot", action="store_true", help="只重畫比較圖")
    args = parser.parse_args()

    WARMUP, REPEATS = args.warmup, args.repeats
    depths = [int(d) for d in args.depths.split(",")]

    if args.plot:
        _plot(depths)
        return

    ns = list(range(args.min_qubits, args.max_qubits + 1, 4))
    bk = available_backends()

    if args.all:
        args.cpu = bk["cpu"]
        args.naive_gpu = bk["naive_gpu"]
        args.cudaq = bk["cudaq"]
        args.ibm = bk["ibm"]

    print("=" * 68)
    print("quantum_stack_benchmark（技術審查版）— 四層後端期望值計算")
    print("=" * 68)
    print(f"qubits: {ns}, depths: {depths}, warmup={WARMUP}, repeats={REPEATS}")
    print(f"可用層: {[k for k, v in bk.items() if v]}")

    layers = {
        "cpu": (backend_cpu, run_cpu),
        "naive_gpu": (backend_naive_gpu, run_naive_gpu),
        "cudaq": (backend_cudaq, run_cudaq),
        "ibm": (backend_ibm, run_ibm),
    }
    enabled = [k for k, flag in
               [("cpu", args.cpu), ("naive_gpu", args.naive_gpu),
                ("cudaq", args.cudaq), ("ibm", args.ibm)] if flag]

    rows = []
    for key in enabled:
        info_fn, run_fn = layers[key]
        device, target, precision = "", "", "complex128"
        try:
            name, device, target = info_fn()
        except Exception as e:
            print(f"[{key}] 初始化失敗: {e}")
            continue
        if args.verify_target or args.all:
            print(f"\n[{key}] device={device}  target={target}")
        for depth in depths:
            for n in ns:
                if key == "ibm" and n != args.min_qubits:
                    continue  # 省額度：IBM 只跑最小 n
                print(f"  [{key:9s}] n={n:2d} depth={depth:2d} ...", end=" ", flush=True)
                runtime, exp = run_fn(n, depth)
                rows.append((key, device, target, precision, n, depth, runtime, exp))
                print(f"runtime={runtime if not np.isnan(runtime) else 'n/a':>8}  <Z0>={exp:.4f}")

    if not rows:
        print("沒有執行任何層。")
        return

    OUT_DIR.mkdir(exist_ok=True)
    # 每個 backend 存自己的 CSV（跨環境執行時不會互相覆寫）
    for key in enabled:
        key_rows = [r for r in rows if r[0] == key]
        np.savetxt(
            OUT_DIR / f"stack_{key}.csv",
            np.array(key_rows, dtype=object),
            header=CSV_HEADER,
            delimiter=",",
            comments="",
            fmt="%s",
        )
        print(f"[data] saved: stack_{key}.csv ({len(key_rows)} rows)")
    _plot(depths)


def _plot(depths: list[int]):
    import csv as _csv

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    layers = {
        "cpu": ("CPU simulator", "#1f77b4", "o-"),
        "naive_gpu": ("naive GPU", "#d62728", "s-"),
        "cudaq": ("CUDA-Q / cuStateVec", "#2ca02c", "^-"),
        "ibm": ("IBM QPU", "#9467bd", "d-"),
    }

    # 合併所有已存在層的 CSV
    data = []
    for key in layers:
        csv_path = OUT_DIR / f"stack_{key}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            data += list(_csv.DictReader(f))
    if not data:
        print("無 stack_*.csv，先跑 benchmark")
        return

    ncol = len(depths)
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 4.5), squeeze=False)
    for j, depth in enumerate(depths):
        ax = axes[0][j]
        plotted = False
        for key, (label, color, marker) in layers.items():
            pts = [(float(r["qubits"]), float(r["runtime_s"]))
                   for r in data if r["backend"] == key and int(r["depth"]) == depth]
            if not pts:
                continue
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker, color=color, label=label)
            plotted = True
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0e}"))
        ax.set_xlabel("Qubits N")
        ax.set_title(f"depth={depth}")
        if j == 0:
            ax.set_ylabel("Expectation time (s, log)")
        if plotted:
            ax.legend(fontsize=8)
    fig.suptitle("Quantum Stack Benchmark - same HEA circuit (IBM shown as semantics only)")
    plt.tight_layout()
    png = OUT_DIR / "stack_benchmark.png"
    plt.savefig(png, dpi=130)
    plt.close()
    print(f"[chart] saved: {png}")


if __name__ == "__main__":
    main()
