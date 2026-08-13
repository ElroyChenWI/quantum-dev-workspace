# Quantum Dev Workspace

[English](README.md)

這是一個可重現的量子軟體開發工作區，用於比較本地模擬、GPU 加速模擬、CUDA-Q，以及 IBM Quantum Runtime 的執行流程。

專案分成兩條主線：

- **演算法主線：** 使用 Qiskit、PennyLane、CUDA-Q 實作同一個 H2 VQE 問題，共用 Hamiltonian 與 ansatz 定義。
- **後端主線：** 使用同一個 hardware-efficient ansatz (HEA) 電路，比較 CPU 模擬、naive GPU 模擬、CUDA-Q/cuStateVec 加速，以及 IBM Quantum 真實 QPU 執行。

這個專案不宣稱實用量子優勢。它的重點是展示一個 backend-agnostic 的量子軟體流程：先在本地驗證，再比較不同模擬器實作，必要時使用 GPU 加速，並將選定電路提交到真實量子硬體觀察含雜訊的執行結果。

## 重點結果

| 項目 | 證據 |
|---|---|
| 跨框架 VQE | Qiskit、PennyLane、CUDA-Q 都收斂到 H2 參考基態能量 `-1.857275 Ha`。 |
| 真實硬體執行 | IBM Quantum Runtime 已用於固定參數 Estimator 檢查，以及 `ibm_kingston` 156-qubit backend 上的小型 hardware-in-the-loop VQE trajectory。 |
| 模擬規模實驗 | CPU statevector 實驗展示 qubit 數增加時的指數成本。 |
| 後端 benchmark | 在 `n=24`、`depth=3` 時，本地測得 CUDA-Q/cuStateVec 單次 expectation evaluation 為 `0.19 s`，CPU 模擬為 `72.8 s`。 |
| 自適應路由 | resource-aware router 會讀取 benchmark CSV，根據 runtime、memory、accuracy mode 與預算建議 CPU、CUDA-Q 或 IBM execution semantics。 |

## 快速開始

Windows 本地環境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python demo.py
```

`demo.py` 會執行 Qiskit 與 PennyLane VQE，並產生收斂比較圖。

GPU 與 CUDA-Q 流程需要 WSL2、Linux 或 Docker。請參考 [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md)。

## H2 VQE

VQE 實驗使用共用的 2-qubit H2 Hamiltonian、4 參數 ansatz 與 COBYLA optimizer。問題定義固定，框架差異只在執行後端。

| Engine | 環境 | 結果 | 備註 |
|---|---|---:|---|
| Exact diagonalization | 本地 CPU | `-1.857275 Ha` | 參考值 |
| Qiskit | 本地 CPU | `-1.857275 Ha` | VQE 收斂，誤差約 `1e-13 Ha` |
| PennyLane | 本地 CPU | `-1.857275 Ha` | VQE 收斂，誤差約 `1e-13 Ha` |
| CUDA-Q | WSL2 + NVIDIA RTX 2060 | `-1.857275 Ha` | GPU target，誤差約 `2e-7 Ha` |

![VQE convergence comparison](quantum_vqe/outputs/vqe_comparison.png)

### IBM Quantum 固定參數檢查

IBM 結果是使用同一個 H2 ansatz、observable 與參數向量 `[0.1, 0.1, 0.05, 0.05]` 所做的單點 Estimator evaluation。

| 來源 | Energy / Expectation |
|---|---:|
| 本地理想 statevector，同參數 | `-1.047914 Ha` |
| IBM Quantum `ibm_kingston`，同參數 | `-1.088185 Ha` |
| 差異 | 約 `0.04 Ha` |

這個差異應解讀為真實硬體 noise、shot statistics、transpilation/layout，以及未做 error mitigation 下的實機行為。它不是與最佳化後 VQE 基態能量的比較。

### IBM Hardware-in-the-Loop VQE

為了區分單點 Estimator 與真正的 VQE loop，專案也加入了一次小型 hardware-in-the-loop VQE：

```text
本地 COBYLA optimizer -> IBM Runtime Estimator energy(theta) -> 更新參數
```

這次在 `ibm_kingston` 上完成 10 次 hardware energy evaluations。第 11 次 evaluation 長時間 queued，已取消；前 10 筆已足以呈現 optimizer 在真硬體 energy landscape 上的下降 trajectory。

| 指標 | 數值 |
|---|---:|
| 初始 IBM evaluation | `-1.056036 Ha` |
| 最佳 IBM evaluation | `-1.526744 Ha` |
| 已完成硬體 evaluations | `10` |
| 最佳 evaluation index | `8` |

![IBM hardware-in-the-loop VQE trajectory](quantum_vqe/outputs/ibm_vqe_trajectory.png)

這應解讀為小型 noisy-hardware VQE trajectory，不是完整收斂的 chemistry-grade 結果。主要時間成本來自 sequential QPU queue 與 job latency。

## 指數規模實驗

規模實驗使用 CPU simulator 評估較大的 HEA 電路，觀察 statevector 模擬成本如何隨 qubit 數增加。

| Qubits | 單次 evaluation 時間 | 近似 statevector 大小 |
|---:|---:|---:|
| 16 | `0.33 s` | `2 MB` |
| 20 | `5.8 s` | `32 MB` |
| 22 | `19.9 s` | `128 MB` |
| 24 | `26.6 s` at depth 1 | `512 MB` |
| 約 30 | 外推為分鐘級 | `16 GB` |

![Scaling experiment](quantum_vqe/outputs/scaling_plot.png)

這個結果說明，在 classical simulation 的範圍內，實務擴展路徑通常包括 GPU 加速、專用模擬器、tensor-network 或近似方法、分散式計算，以及真實 QPU 執行。

## Quantum Stack Benchmark

Benchmark 使用同一個 HEA 電路：

```text
RY/RZ on all qubits -> CNOT chain -> measure <Z0>
```

| 層級 | 後端 | 角色 |
|---|---|---|
| L1 | Qiskit CPU statevector | 正確性基準 |
| L2 | CuPy naive GPU statevector | 教學用 GPU baseline，不是 production simulator |
| L3 | CUDA-Q `nvidia` target / cuStateVec | GPU 加速量子模擬後端 |
| L4 | IBM Quantum QPU | 真實硬體語義比較，不比較 runtime |

測量方法：

- `--verify-target` 記錄實際 device 與 backend target。
- GPU timing 包含 explicit synchronization，避免低估 async kernel 時間。
- 排除 warmup，回報重複測量的 median。
- 結果以 CSV 輸出：`backend, device, target, precision, qubits, depth, runtime_s, expectation`。
- 已處理 Qiskit Pauli endianness，確保 CPU、naive GPU、CUDA-Q 都量測同一個 qubit-0 observable。

WSL2 + NVIDIA RTX 2060 實測：

| Depth | CPU | naive GPU | CUDA-Q/cuStateVec | CPU / CUDA-Q |
|---:|---:|---:|---:|---:|
| 1 | `35.3 s` | `1.47 s` | `0.13 s` | 約 `265x` |
| 3 | `72.8 s` | `4.36 s` | `0.19 s` | 約 `389x` |

![Quantum stack speedup summary](quantum_vqe/outputs/stack_summary.png)

![Full stack benchmark](quantum_vqe/outputs/stack_benchmark.png)

12 個共同測點的數值交叉驗證：

| 比較 | 最大 `<Z0>` 差異 |
|---|---:|
| CPU vs naive GPU | `4.6e-14` |
| CPU vs CUDA-Q | `3.8e-7` |

IBM Quantum 另行處理，因為 queue time 與含雜訊的 QPU execution 不應直接與 simulator runtime 比較。

## Resource-Aware Router

專案新增了一個面向變分量子工作流的自適應執行路由器。它會讀取 benchmark CSV，擬合簡單 runtime model，估算 statevector memory，並在明確條件下建議執行後端。

```powershell
python quantum_vqe/quantum_router.py --qubits 8 --depth 1 --accuracy exact --time-budget 1
python quantum_vqe/quantum_router.py --qubits 24 --depth 3 --accuracy exact --time-budget 10
python quantum_vqe/quantum_router.py --qubits 32 --depth 3 --accuracy hardware --allow-ibm
```

| 模式 | 行為 |
|---|---|
| `exact` | 只選 CPU / CUDA-Q 這類 noiseless local simulators。 |
| `estimate` | 選最快可行的本地後端；只有明確允許時才會建議 IBM。 |
| `hardware` | 只有在要求 real-QPU semantics 且提供 `--allow-ibm` 時才選 IBM。 |

這是一個 resource-aware execution policy prototype，不是通用量子排程器。IBM 被刻意 gate 起來，因為 QPU 結果含雜訊且受 queue time 影響。

## 複現方式

本地 CPU：

```powershell
python demo.py
python quantum_vqe/scale_experiment.py
python quantum_vqe/quantum_stack_benchmark.py --cpu --verify-target
python quantum_vqe/quantum_stack_benchmark.py --plot --depths 1,3
python quantum_vqe/quantum_router.py --qubits 24 --depth 3 --accuracy exact --time-budget 10
```

WSL2 CUDA-Q：

```bash
cd /mnt/d/Elroy/Quant_DEV
python quantum_vqe/run_cudaq.py
python quantum_vqe/quantum_stack_benchmark.py --naive-gpu --cudaq --verify-target --depths 1,3 --max_qubits 24
```

IBM Quantum：

```powershell
python quantum_vqe/run_ibm_cloud.py --list
python quantum_vqe/run_ibm_cloud.py --backend ibm_kingston
python quantum_vqe/run_ibm_vqe.py --backend ibm_kingston --maxiter 12
python quantum_vqe/run_ibm_vqe_batched.py --backend ibm_kingston --rounds 20 --session-max-time 2h
```

需要在 `.env` 或環境變數設定 `QISKIT_IBM_TOKEN`。請勿提交真實 token。

## 限制與注意事項

- 本地 benchmark 數字會受硬體與系統負載影響。
- naive GPU simulator 是刻意簡化的 baseline，只用於教育性比較。
- CUDA-Q/cuStateVec 仍屬 classical simulation，不是真實量子硬體。
- IBM Quantum 結果反映含雜訊的硬體執行，不應直接與 simulator runtime 比較。
- 固定參數 IBM 結果不是 VQE 收斂結果。
- IBM hardware-in-the-loop VQE 是小型 noisy run，不應解讀為完整收斂的 chemistry 計算。
- 若要讓硬體 VQE 更接近「一次跑完」，請優先使用 `run_ibm_vqe_batched.py`。它使用 Runtime Session，並在每一輪把多個候選參數打包成同一個 Estimator job。
