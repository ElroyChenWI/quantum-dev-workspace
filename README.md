# Quantum Dev Workspace

跨框架量子開發環境 · Cross-Framework Quantum Development Environment

整合 **Qiskit / PennyLane / Cirq / CUDA-Q** 的量子開發環境，從基礎教學到進階專案，核心為「同一個物理問題、跨三個框架執行」的 VQE 專案。

A quantum development environment integrating **Qiskit / PennyLane / Cirq / CUDA-Q**, from fundamentals to advanced projects. The centerpiece is a VQE project that runs one physics problem across three frameworks.

## 我在這個 repo 做了什麼 / What This Repo Demonstrates

這個 repo 不是「環境測試」，而是一個**完整可驗證的量子計算專案**：從一個物理問題（H₂ 基態）出發，跨三套框架收斂到同一精確值，再沿「本機 CPU → GPU → 真實量子晶片」的完整堆疊一路實測到頂。

This repo is **not an environment test — it is a complete, verifiable quantum computing project**: one physics problem (H₂ ground state) solved across three frameworks, then pushed all the way up the stack from local CPU → GPU → real quantum hardware.

| # | 成果 / Result | 證據 / Evidence |
|---|---------------|-----------------|
| 1 | 同一物理問題跨 4 引擎執行，全收斂到精確基態能量 **-1.857275 Ha**（誤差 < 1e-13）<br>Same problem across 4 engines, all converge to **-1.857275 Ha** (<1e-13 error) | [VQE 專案](#vqe-跨框架專案主角-flagship-one-problem-three-frameworks) |
| 2 | 真實 **156-qubit** 量子處理器實測（IBM `ibm_kingston`），量化硬體雜訊 ~0.04 Ha<br>Real 156-qubit QPU (IBM `ibm_kingston`), hardware noise quantified ~0.04 Ha | [真硬體驗證](#ibm-quantum-真硬體單點驗證) |
| 3 | 實證模擬的「指數牆」，量化為何必須上雲端 / 真硬體<br>Empirically demonstrated the exponential wall, quantifying why cloud/hardware is necessary | [規模實驗](#規模實驗指數牆) |
| 4 | 四層後端平台 Benchmark（技術審查級測量方法），n=24/depth=3 時 CUDA-Q 比 CPU **快 ~389×**<br>4-layer backend benchmark (technical-review measurement), CUDA-Q **~389× faster** than CPU at n=24/depth=3 | [平台堆疊 Benchmark](#平台堆疊-benchmark四層) |

這四張牌合起來：**「我會建環境、我會寫演算法、我會量性能、我真的接過真硬體。」**
Together: **“I can build the environment, I can write the algorithm, I can measure performance, and I have actually touched real hardware.”**

## 一鍵體驗 / One-Command Demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python demo.py
```

`demo.py` 依序執行 Qiskit / PennyLane VQE，最後畫出兩框架的收斂比較圖，預期兩者都收斂到 **-1.857275 Ha**（誤差 < 1e-13）。

`demo.py` runs the Qiskit and PennyLane VQE in sequence, then plots the convergence comparison. Both converge to **-1.857275 Ha** within < 1e-13.

> 完整複現所有結果 / Reproduce everything:
> ```powershell
> python demo.py                                  # 一鍵體驗 / one-command demo
> python quantum_vqe/quantum_stack_benchmark.py --cpu --verify-target   # L1 benchmark (Windows)
> # WSL: python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq --verify-target  # L2/L3
> python quantum_vqe/scale_experiment.py         # 指數牆 / exponential wall
> python quantum_vqe/plot_comparison.py          # 收斂比較圖 / convergence plot
> ```

## VQE 跨框架專案（主角）/ Flagship: One Problem, Three Frameworks

用**同一個 H₂ 分子問題**、**同一個 ansatz**、**同一個最佳化器**，在 Qiskit / PennyLane / CUDA-Q 三個框架實作 VQE，全部收斂到同一個精確基態能量；並以同一組參數提交到 IBM 真硬體量化雜訊。換框架只是換「量子執行引擎」，問題定義完全共用。

The same H₂ molecule problem, the same ansatz, and the same optimizer, implemented in Qiskit / PennyLane / CUDA-Q. All converge to the exact same ground-state energy — and the same parameters were submitted to IBM real hardware to quantify noise. Switching framework only swaps the quantum execution engine.

| 引擎 / Engine | 執行環境 / Environment | 基態能量 / Energy (Ha) | 說明 / Notes |
|------|------|------|------|
| 精確值 / Exact | 對角化 / diagonalization | -1.857275 | 參考基準 / Reference |
| Qiskit | 本機 CPU / Local CPU | -1.857275 | VQE 收斂（誤差 ~1e-13）|
| PennyLane | 本機 CPU / Local CPU | -1.857275 | VQE 收斂（誤差 ~1e-13）|
| CUDA-Q | WSL2 + NVIDIA GPU (RTX 2060) | -1.857275 | VQE 收斂（誤差 ~2e-7）|
| **IBM Quantum** | **真實 156-qubit 處理器** / real QPU | **-1.088185**（同參數）| 硬體雜訊 ~0.04，見下方 IBM 段 |

![VQE 收斂比較 / VQE convergence comparison](quantum_vqe/outputs/vqe_comparison.png)

> 詳見 / See [quantum_vqe/README.md](quantum_vqe/README.md)

### IBM Quantum 真硬體（單點驗證）

在**同一組參數** `[0.1, 0.1, 0.05, 0.05]` 下，比較本機理想模擬與 IBM 真實 156-qubit 處理器：

| 執行 | 能量 (Ha) | 差異 |
|------|----------|------|
| 本機理想模擬（同參數）| -1.047914 | — |
| **IBM Quantum 真硬體（ibm_kingston）** | **-1.088185** | **~0.04（= 純硬體雜訊）** |

> 這個 ~0.04 的差異是「**同一電路、同一參數**」下，理想模擬 vs 真硬體的**雜訊**。
> （對照：完整 VQE 收斂後的基態能量是 -1.857275 Ha。）

### 實測：IBM Quantum 真硬體

H2 的同一個 ansatz / observable 已在 **IBM Quantum 真實 156-qubit 處理器（ibm_kingston）** 上執行 Estimator job。
同一組參數在本機理想 statevector 的 expectation value 為 **-1.047914 Ha**，真硬體回傳 **-1.088185 Ha**。
差異 **~0.04 Ha** 主要呈現真實量子硬體的 noise、shot statistics、transpilation/layout 與未做 error mitigation 的實機效應。
詳見 / See [docs/IBM_CLOUD_AUDIT.md](docs/IBM_CLOUD_AUDIT.md)。

## 規模實驗：指數牆 / Scale Experiment: The Exponential Wall

同一個 HEA 電路在 CPU 模擬器上逐步加 qubit，實測每次 expectation 的計算時間與記憶體——**用數據證明模擬為什麼無法擴展**。

The same HEA circuit on a CPU simulator, adding qubits to measure time & memory per evaluation — **data that proves why simulation cannot scale**. 實測結果 / Measured:

| N | 每次評估時間 / time per eval | 狀態向量 / statevector |
|----|------|------|
| 16 | 0.33 s | 2 MB |
| 20 | 5.8 s | 32 MB |
| 22 | **19.9 s** | 128 MB |
| 24 | **26.6 s**（depth=1）| 512 MB |
| ~30 | 外推 ~數分鐘 / extrapolated minutes | **16 GB** |

![規模實驗 / Scale experiment](quantum_vqe/outputs/scaling_plot.png)

> 這不是「大小問題」，是**指數牆**——模擬時間與記憶體隨 qubit 數指數成長，
> 實務上的擴展路徑包括 GPU 加速、專用模擬器、近似方法，以及真實 QPU 執行。
> This is not a “size problem” — it is the **exponential wall**. Simulation time & memory grow exponentially with qubit count; practical scaling paths include GPU acceleration, specialized simulators, approximation methods, and real QPU execution.

## 平台堆疊 Benchmark（四層）/ 4-Layer Stack Benchmark

用**同一個 hardware-efficient ansatz（HEA）電路**（RY+RZ 全 qubit → CNOT 鏈 → 量 ⟨Z₀⟩），在四層後端上量測同一計算的時間——展示「同樣的量子計算，在不同平台上差多少」。

The **same HEA circuit** (RY+RZ all qubits → CNOT chain → measure ⟨Z₀⟩), timed on four backends — showing how much the *same* quantum computation costs on different platforms.

| 層 / Layer | 後端 / Backend | 目的 / Purpose |
|----|------|------|
| L1 | CPU simulator（Qiskit Statevector）| 正確性基準 / correctness baseline |
| L2 | naive GPU（手寫 CuPy，教育用平行化）| 一般 GPU 平行化的限制 / limits of naive GPU parallelism |
| L3 | CUDA-Q / cuStateVec（`nvidia` target）| 專用量子模擬 kernel 的優勢 / advantage of dedicated quantum kernels |
| L4 | IBM Quantum（真實 QPU）| 真硬體執行（語義比較，非秒數） / real QPU (semantic comparison, not runtime) |

**測量方法（技術審查級）/ Measurement methodology (technical-review level):**
- `--verify-target`：執行時印出真實 device / target（如 `NVIDIA GeForce RTX 2060`、`target=nvidia (1 QPU)`），杜絕「跑錯後端還不自知」
- GPU 計時強制 **synchronize**：避免 async kernel 讓計時失真
- **warmup + median**：`--warmup 1 --repeats 2`，重複取中位數，排除冷啟動與抖動
- **多深度掃描**：`--depths 1,3,6,10`，同時看電路深度對各平台成本的影響
- **統一 CSV**：`backend, device, target, precision, qubits, depth, runtime_s, expectation`
- **交叉驗證**：CPU / naive GPU / CUDA-Q 三層的 ⟨Z₀⟩ 完全一致（Qiskit Pauli 已處理 endianness，三者量同一個 qubit 0）→ 三個實作互相驗證正確

**實測結果（RTX 2060，n=24，warmup=1 + repeats=2 中位數）/** Measured results:

| Depth | CPU (L1) | naive GPU (L2) | CUDA-Q (L3) | CPU / CUDA-Q |
|-------|----------|----------------|-------------|--------------|
| 1 | 35.3 s | 1.47 s | **0.13 s** | ~265× |
| 3 | 72.8 s | 4.36 s | **0.19 s** | **~389×** |

![平台堆疊速度總結 / Stack speedup summary](quantum_vqe/outputs/stack_summary.png)

![平台堆疊完整比較 / Full stack benchmark](quantum_vqe/outputs/stack_benchmark.png)

> Cross-check: across 12 shared points, max CPU-vs-naive-GPU `<Z0>` difference is `4.6e-14`; max CPU-vs-CUDA-Q difference is `3.8e-7`.

> L4 IBM 為語義比較：真硬體回傳的是含雜訊的 expectation value，**不與模擬秒數直接比**（詳見上方 IBM 真硬體段）。
> L4 IBM is a semantic comparison: real hardware returns a noise-corrupted expectation, **not directly comparable to simulation runtime** (see IBM section above).
>
> 執行 / Run：`python quantum_vqe/quantum_stack_benchmark.py --help`（L1 在 Windows、L2/L3 在 WSL 的 CUDA-Q 環境、`--plot` 合併所有 `stack_*.csv`）。

## 使用的框架 / Frameworks

| 框架 / Framework | 用途 / Purpose | 後端 / Backend |
|------|------|------|
| [Qiskit](https://qiskit.org/) | 量子電路、量子演算法 / circuits, quantum algorithms | 本機模擬 + IBM Quantum 雲端實測（156-qubit）/ local simulation + IBM Quantum cloud verified (156-qubit) |
| [PennyLane](https://pennylane.ai/) | 可微分量子程式、量子機器學習 / differentiable programs, QML | 多種模擬器 + 硬體 / simulators + hardware |
| [Cirq](https://quantumai.google/cirq) | 量子電路研究 / circuit research | 本機模擬器 / local simulator |
| [CUDA-Q](https://developer.nvidia.com/cuda-q) | GPU 高效能模擬、混合量子-經典 / GPU simulation, hybrid QC | NVIDIA GPU（需 WSL2 / Linux / Docker）|

## 教學範例 / Tutorial Examples

| 範例 / Example | 內容 / Content |
|------|------|
| `examples/00_entanglement_vs_independent.py` | 糾纏 vs 不糾纏對照（互相資訊）/ Entanglement vs independent (mutual information) |
| `examples/01_quantum_computing_basics.py` | Bell / GHZ 態、機率分布 / Bell & GHZ states, probabilities |
| `examples/02_quantum_machine_learning.py` | PennyLane 梯度訓練 + Qiskit VQC / PennyLane gradient training + Qiskit VQC |
| `examples/03_quantum_simulation.py` | Trotter 演化 + Cirq 密度矩陣 / Trotter evolution + Cirq density matrix |

## 專案結構 / Project Structure

```
Quant_DEV/
├── README.md               # 本文件 / This file
├── requirements.txt        # Windows 依賴 / Windows dependencies
├── requirements-cudaq.txt  # CUDA-Q 依賴（WSL2/Linux/Docker）
├── demo.py                 # 一鍵體驗 / one-command demo
├── examples/               # 基礎教學範例 / tutorial examples
├── quantum_vqe/            # ★ 旗艦專案 / flagship project
│   ├── h2_hamiltonian.py   #   H₂ 問題定義（唯一來源，全部引擎共用）
│   ├── run_qiskit.py       #   VQE: Qiskit
│   ├── run_pennylane.py    #   VQE: PennyLane
│   ├── run_cudaq.py        #   VQE: CUDA-Q（GPU）
│   ├── run_ibm_cloud.py    #   IBM Quantum 真硬體提交
│   ├── scale_experiment.py #   規模實驗：指數牆
│   ├── quantum_stack_benchmark.py  # 四層平台堆疊 Benchmark
│   ├── plot_comparison.py  #   收斂比較圖
│   └── outputs/            #   所有結果 CSV + 圖表
├── src/quant_dev/          # 可重用程式碼 / reusable code
├── docs/CUDA-Q_SETUP.md    # CUDA-Q 安裝指引 / CUDA-Q setup guide
├── docs/IBM_CLOUD_AUDIT.md # IBM 雲端整合審計 / IBM cloud integration audit
└── outputs/                # 執行結果圖表 / generated charts
```

## CUDA-Q

CUDA-Q 目前**不支援 Windows 原生執行**，請使用 **WSL2** 或 **Docker**。詳細步驟見 [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md)。

CUDA-Q does **not** run natively on Windows; use **WSL2** or **Docker**. See [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md).

## 之後連上真實量子電腦 / Connect to Real Quantum Hardware

- **IBM Quantum**：已在 **真實 156-qubit 處理器**上實測（見上方 IBM 段）。到 [IBM Quantum Platform](https://quantum.ibm.com/) 註冊、取得 API token，設定 `QISKIT_IBM_TOKEN` 即可重現。
  Already verified on a **real 156-qubit processor** (see IBM section above). Register at [IBM Quantum Platform](https://quantum.ibm.com/), set `QISKIT_IBM_TOKEN`, and reproduce.
- **CUDA-Q GPU**：已於 **WSL2 + NVIDIA RTX 2060** 完成 GPU 模擬（VQE + Benchmark 皆實測）。CUDA-Q 不支援 Windows 原生執行，見 [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md)。
  Already verified on **WSL2 + NVIDIA RTX 2060** (VQE + benchmark). CUDA-Q does not run natively on Windows; see [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md).
- 本專案預設先以本機模擬器開發，程式碼保留接雲端/GPU 的彈性（同一 `h2_hamiltonian.py` 四處共用）。
  Developed on local simulators by default, with cloud/GPU-ready code (one shared `h2_hamiltonian.py`).

## 參考資源 / Resources

- [Qiskit 官方教學 / Qiskit Learning](https://learning.quantum.ibm.com/)
- [PennyLane 文件 / PennyLane Docs](https://docs.pennylane.ai/)
- [Cirq 文件 / Cirq Docs](https://quantumai.google/cirq)
- [CUDA-Q 文件 / CUDA-Q Docs](https://docs.nvidia.com/cuda-quantum/latest/)
