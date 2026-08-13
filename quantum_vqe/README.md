# VQE 跨框架專案 — H₂ 分子基態能量（Qiskit / PennyLane / CUDA-Q）

用**同一個物理問題**（H₂ 分子的基態能量）、**同一個 ansatz**、**同一個最佳化器**，
在三個量子框架上各自實作 VQE，驗證它們全部收斂到同一個精確答案。

> 這是一個展示「量子開發環境全面性」的小專案：換框架只是換「量子執行引擎」，
> 問題定義與結構完全共用。

## 問題背景

VQE（變分量子特徵求解器）是量子化學最成熟的演算法：

1. 用一個參數化量子電路（ansatz）準備一個猜測的量子態 $|\psi(\theta)\rangle$
2. 測量哈密頓量 $\langle \psi(\theta) | H | \psi(\theta) \rangle$（能量期望值）
3. 經典優化器調整參數 $\theta$，讓能量越來越低
4. 收斂到的最低值 = 基態能量（基態能量越低，系統越穩定）

對 H₂：基態能量 **-1.857275 Ha**（哈特里）。

## 專案結構

```
quantum_vqe/
├── README.md               # 本文件
├── h2_hamiltonian.py       # ★ 共用核心：H2 2-qubit 哈密頓量 + 精確解
├── run_qiskit.py           # Qiskit 版（StatevectorEstimator）— Windows 可跑
├── run_pennylane.py        # PennyLane 版（default.qubit）— Windows 可跑
├── run_cudaq.py            # CUDA-Q 版（需 WSL2 / Docker）
├── plot_comparison.py      # 把三框架收斂曲線畫在同一張圖比較
├── scale_experiment.py     # 規模實驗：加大 qubit 數，直到撞到指數牆
├── quantum_stack_benchmark.py  # 量子平台堆疊四層 benchmark（CPU/naive GPU/CUDA-Q/IBM）
├── wsl_test_gpu.py         # CUDA-Q GPU 煙霧測試（Bell 態）
└── outputs/                # 收斂資料 CSV + 圖表
```

## 快速執行（Windows）

```powershell
# 在專案根目錄執行（d:\Elroy\Quant_DEV）
.\.venv\Scripts\Activate.ps1

cd quantum_vqe
python run_qiskit.py        # Qiskit VQE
python run_pennylane.py     # PennyLane VQE
python plot_comparison.py   # 畫比較圖
```

預期結果（三個框架都收斂到精確值，誤差 < 1e-12 Ha）：

| 框架 | 能量 (Ha) | 誤差 |
|------|----------|------|
| 精確值 | -1.857275 | — |
| Qiskit | -1.857275 | ~1e-13 |
| PennyLane | -1.857275 | ~1e-13 |
| CUDA-Q | -1.857275 | ~2e-7（GPU）|

## CUDA-Q 版（WSL2 + NVIDIA GPU）

CUDA-Q 不支援 Windows 原生執行，需在 WSL2 / Docker 內執行。
本專案已在 **WSL2 + NVIDIA RTX 2060** 上驗證（`nvidia` GPU target），收斂到 -1.857275 Ha。

執行（WSL 內，需有 python 3.12 的 `cudaq` 環境）：

```bash
cd /mnt/d/Elroy/Quant_DEV/quantum_vqe
# 使用 cudaq 環境的 python（例如 conda 環境 cudaq）
python run_cudaq.py
```

CUDA-Q 的好處是可吃 NVIDIA GPU 加速；本範例的程式結構與 Qiskit/PennyLane 完全相同，
只差在 `cudaq.observe` 這個「執行引擎」呼叫。

> 若在 Windows 上執行，會提示 CUDA-Q 未安裝（需 WSL2/Docker），並指引到 `docs/CUDA-Q_SETUP.md`。

## 規模實驗：找到指數牆

從 H2（2 qubit）出發，改用可擴展的 1D Heisenberg 鏈哈密頓量，把 qubit 數一路加大，
量測單次能量評估的時間與狀態向量記憶體，親眼看到經典模擬的指數成長。

```powershell
python scale_experiment.py             # 預設跑到 24 qubit
python scale_experiment.py --max_qubits 28   # 更接近牆（小心記憶體）
```

預期觀察：每加 2 個 qubit，單次評估時間約 ×5，log 尺規下是一條直線；
狀態向量記憶體以 2^N 成長（16 bytes × 2^N），本機的「牆」約在 N≈28–30。

![規模實驗：指數牆](outputs/scaling_plot.png)

> 這就是「為什麼要上雲端/真硬體」的實證——不是大小問題，是指數牆的問題。

## 量子平台堆疊 Benchmark（四層）

用**同一個 hardware-efficient ansatz（HEA）電路**，在四層後端上量測期望值計算時間，
展示平台堆疊的差異。

| 層 | 後端 | 目的 |
|----|------|------|
| L1 | CPU simulator（Qiskit）| 正確性基準 |
| L2 | naive GPU（手寫 cupy）| 一般 GPU 平行化的限制 |
| L3 | CUDA-Q / cuStateVec（nvidia）| 專用量子模擬 kernel 的優勢 |
| L4 | IBM Quantum | 真實 QPU 執行 |

執行：

```powershell
# Windows（L1）
python quantum_vqe/quantum_stack_benchmark.py --cpu

# WSL（L2/L3）
python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq

# 合併比較圖（讀取所有 stack_*.csv）
python quantum_vqe/quantum_stack_benchmark.py --plot
```

實測結果（depth=1 與 depth=3，RTX 2060，warmup=1、repeats=2 取中位數）：

| N | CPU (d1) | naive GPU (d1) | CUDA-Q (d1) | CPU (d3) | naive GPU (d3) | CUDA-Q (d3) |
|---|----------|----------------|-------------|----------|----------------|-------------|
| 16 | 0.13 s | 0.14 s | 0.14 s | 0.17 s | 0.20 s | 0.09 s |
| 20 | 2.19 s | 0.14 s | 0.13 s | 4.10 s | 0.33 s | 0.09 s |
| 24 | **35.3 s** | 1.47 s | **0.13 s** | **72.8 s** | 4.36 s | **0.19 s** |

![平台堆疊速度總結](outputs/stack_summary.png)

> 重點 1：**CPU / naive GPU / CUDA-Q 三層算出的 ⟨Z₀⟩ 完全一致**（同一 observable，三層交叉驗證正確；
>         Qiskit Pauli 已處理 endianness，量到同一個 qubit 0）。
> 重點 2：depth=3、n=24 時，CUDA-Q 比 CPU **快約 389 倍**，比 naive GPU 快約 23 倍。
> 重點 3：測量方法已專業化——`--verify-target` 印出真實 device/target、
>         GPU 計時強制 sync（避免 async 造成時間偏低）、warmup + median、多深度掃描。
> 註：CPU 計時為單機實測，受系統負載影響約 ±30%（n=24/depth=3 曾測得 90.6 s）。

測量方法與輸出：

```powershell
python quantum_vqe/quantum_stack_benchmark.py --cpu --verify-target      # L1（Windows）
python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq --verify-target   # L2/L3（WSL）
python quantum_vqe/quantum_stack_benchmark.py --plot                     # 合併圖
python quantum_vqe/quantum_stack_benchmark.py --all --verify-target --depths 1,3,6,10 --max_qubits 24
```

統一輸出格式（`outputs/stack_<backend>.csv`）：
`backend, device, target, precision, qubits, depth, runtime_s, expectation`

> 圖表：`outputs/stack_summary.png`（n=24 摘要）與 `outputs/stack_benchmark.png`（依 depth 分 subplot）

## IBM Quantum 雲端實測

H2 已實際提交到 **IBM Quantum 真實 156-qubit 處理器**（ibm_kingston）執行：

| 來源 | 能量 (Ha) |
|------|-----------|
| 本機模擬（理想）| -1.047914 |
| IBM 真硬體 | -1.088185 |
| 差異（= 硬體雜訊）| ~0.04 |

執行方式（需先在 `.env` 設定 `QISKIT_IBM_TOKEN`）：

```powershell
python quantum_vqe/run_ibm_cloud.py --list          # 列出可用 backend
python quantum_vqe/run_ibm_cloud.py --backend ibm_kingston   # 跑 H2 能量
```

> 真硬體結果與本機理想值的差異，正是量子硬體雜訊的具體呈現。
> 詳見 / See [docs/IBM_CLOUD_AUDIT.md](../docs/IBM_CLOUD_AUDIT.md)。

## 三框架的對照（跨框架的關鍵）

| 層面 | Qiskit | PennyLane | CUDA-Q |
|------|--------|-----------|--------|
| 量子執行 | `StatevectorEstimator` | `default.qubit` | `cudaq.observe` |
| 哈密頓量 | `SparsePauliOp.from_list` | `qml.dot` | `spin` operator |
| ansatz | 4 參數 `Ry`+`CNOT` | 相同 | 相同 |
| 最佳化器 | scipy COBYLA | scipy COBYLA | scipy COBYLA |
| 精確解對照 | 共用 `h2_hamiltonian.py` | 相同 | 相同 |

三者讀同一份 `PAULI_TERMS`，這就是「跨框架」的工程關鍵：**問題定義一次，引擎任意換**。
