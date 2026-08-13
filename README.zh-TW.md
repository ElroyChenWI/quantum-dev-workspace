# Quantum Dev Workspace

[English](README.md)

這是一個可重現的量子軟體開發 workspace，用來比較本機模擬、GPU 加速模擬、CUDA-Q，以及 IBM Quantum Runtime 的執行流程。

這個 repo 的重點不是宣稱實用量子優勢，而是建立一個專業的量子開發流程：先在本機驗證演算法，再比較不同模擬後端，必要時使用 GPU 加速，最後將選定的電路送到真實 QPU 觀察 noisy hardware 行為。

## 核心內容

| 主題 | 證據 |
|---|---|
| 跨框架 VQE | Qiskit、PennyLane、CUDA-Q 使用同一個 H2 Hamiltonian 與 ansatz，收斂到 `-1.857275 Ha`。 |
| 真實硬體執行 | IBM Quantum Runtime 已用於固定參數 Estimator 檢查，以及 `ibm_kingston` 上的小型 hardware-in-the-loop VQE trajectory。 |
| 模擬器擴展性 | CPU statevector 實驗顯示 qubit 增加時的指數成本。 |
| 後端 benchmark | 在本機量測中，`n=24`、`depth=3` 時 CUDA-Q/cuStateVec 完成一次 expectation evaluation 約 `0.19 s`，CPU 約 `72.8 s`。 |
| 自適應路由 | Resource-aware router 會根據 benchmark CSV、記憶體估計、精度模式與時間限制，建議 CPU、CUDA-Q 或 IBM 執行語意。 |

## 環境自檢

新機器 clone repo 後，建議先跑本機環境自檢。這個程序會記錄 Python 套件、GPU 工具、WSL 狀態，以及小型 CPU/CUDA-Q smoke benchmark。

```powershell
python quantum_vqe/quantum_env_check.py --max-qubits 12 --depth 1
```

輸出檔案：

- `quantum_vqe/outputs/env_profile.json`：機器可讀的本機能力 profile。
- `quantum_vqe/outputs/env_profile.md`：人可讀的環境報告。

雲端/QPU 能力另外檢查：

```powershell
python quantum_vqe/quantum_cloud_check.py --backend ibm_kingston
```

這會輸出 `quantum_vqe/outputs/cloud_profile.json` 與 `quantum_vqe/outputs/cloud_profile.md`，記錄 IBM 帳號是否可用、Runtime client 可取得的 usage/quota 資訊、backend operational 狀態與 pending jobs。profile 不會寫入 token 或帳號識別碼。

router 預設會讀取這兩份 profile：`env_profile.json` 決定 CPU/GPU 本機能力，`cloud_profile.json` 決定 IBM 是否可被推薦。

## 快速開始

Windows 本機環境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python demo.py
```

GPU 與 CUDA-Q workflow 需要 WSL2、Linux 或 Docker。設定方式見 [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md)。

## H2 VQE

H2 VQE 使用共用的 2-qubit Hamiltonian、4 參數 ansatz 與 COBYLA optimizer。不同框架只替換執行引擎，問題定義保持一致。

| Engine | 環境 | 結果 | 備註 |
|---|---|---:|---|
| Exact diagonalization | 本機 CPU | `-1.857275 Ha` | 參考值 |
| Qiskit | 本機 CPU | `-1.857275 Ha` | VQE 收斂，誤差約 `1e-13 Ha` |
| PennyLane | 本機 CPU | `-1.857275 Ha` | VQE 收斂，誤差約 `1e-13 Ha` |
| CUDA-Q | WSL2 + NVIDIA RTX 2060 | `-1.857275 Ha` | GPU target，誤差約 `2e-7 Ha` |

![VQE convergence comparison](quantum_vqe/outputs/vqe_comparison.png)

## IBM Quantum

固定參數 IBM 檢查使用同一個 H2 ansatz、observable 與參數 `[0.1, 0.1, 0.05, 0.05]`。

| 來源 | Energy / Expectation |
|---|---:|
| 本機 ideal statevector，同參數 | `-1.047914 Ha` |
| IBM Quantum `ibm_kingston`，同參數 | `-1.088185 Ha` |
| 差異 | 約 `0.04 Ha` |

這個結果應解讀為 real-hardware behavior，包含 noise、shot statistics、transpilation/layout 與未使用 error mitigation 的影響。它不是與最佳化後 VQE ground-state energy 的直接比較。

repo 也包含 hardware-in-the-loop VQE：

```text
local COBYLA optimizer -> IBM Runtime Estimator energy(theta) -> parameter update
```

已記錄的 IBM run 在 `ibm_kingston` 完成 10 次 hardware energy evaluations，最佳觀測值為 `-1.526744 Ha`。這是小型 noisy-hardware VQE trajectory，不應解讀為已完全收斂的 chemistry-grade 結果。

![IBM hardware-in-the-loop VQE trajectory](quantum_vqe/outputs/ibm_vqe_trajectory.png)

## Quantum Stack Benchmark

Stack benchmark 使用同一個 hardware-efficient ansatz：

```text
RY/RZ on all qubits -> CNOT chain -> measure <Z0>
```

| Layer | Backend | 角色 |
|---|---|---|
| L1 | Qiskit CPU statevector | 正確性基準 |
| L2 | CuPy naive GPU statevector | 教學用 GPU baseline |
| L3 | CUDA-Q `nvidia` target / cuStateVec | GPU 加速量子模擬 |
| L4 | IBM Quantum QPU | 真實硬體語意比較 |

WSL2 + NVIDIA RTX 2060 量測：

| Depth | CPU | naive GPU | CUDA-Q/cuStateVec | CPU / CUDA-Q |
|---:|---:|---:|---:|---:|
| 1 | `35.3 s` | `1.47 s` | `0.13 s` | 約 `265x` |
| 3 | `72.8 s` | `4.36 s` | `0.19 s` | 約 `389x` |

![Quantum stack speedup summary](quantum_vqe/outputs/stack_summary.png)

![Full stack benchmark](quantum_vqe/outputs/stack_benchmark.png)

12 個共同量測點的數值交叉檢查：

| 比較 | 最大 `<Z0>` 差異 |
|---|---:|
| CPU vs naive GPU | `4.6e-14` |
| CPU vs CUDA-Q | `3.8e-7` |

IBM Quantum 另外處理，因為 queue time 與 noisy QPU execution 不能直接和 simulator runtime 比較。

## Resource-Aware Router

`quantum_router.py` 會讀取 benchmark CSV，建立簡單 runtime model，估計 statevector memory，並在明確限制下建議執行後端。

```powershell
python quantum_vqe/quantum_router.py --qubits 8 --depth 1 --accuracy exact --time-budget 1
python quantum_vqe/quantum_router.py --qubits 24 --depth 3 --accuracy exact --time-budget 10
python quantum_vqe/quantum_router.py --qubits 32 --depth 3 --accuracy hardware --allow-ibm
```

router 預設也會讀取 `quantum_vqe/outputs/env_profile.json` 與 `quantum_vqe/outputs/cloud_profile.json`。本機 profile 控制 CPU/GPU availability 與記憶體預算；cloud profile 控制 IBM availability、usage limit 狀態與 backend operational 狀態。

| 模式 | 行為 |
|---|---|
| `exact` | 只選擇 CPU / CUDA-Q 等 noiseless local simulators。 |
| `estimate` | 選擇最快的可行本機後端，只有在明確允許時才可能建議 IBM。 |
| `hardware` | 只有在要求 real-QPU semantics 且提供 `--allow-ibm` 時才選擇 IBM。 |

這是一個 resource-aware execution policy prototype，不是通用量子排程器。IBM 被刻意設為 gated backend，因為 QPU 結果 noisy、queue-limited，且精度語意不同。

## 重現結果

本機 CPU：

```powershell
python demo.py
python quantum_vqe/quantum_env_check.py --max-qubits 12 --depth 1
python quantum_vqe/quantum_cloud_check.py --backend ibm_kingston
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

IBM workflow 需要在 `.env` 或環境變數設定 `QISKIT_IBM_TOKEN`。不要提交真實 token。

## 專案結構

```text
Quant_DEV/
+-- README.md
+-- README.zh-TW.md
+-- requirements.txt
+-- requirements-cudaq.txt
+-- demo.py
+-- examples/
+-- quantum_vqe/
|   +-- h2_hamiltonian.py
|   +-- run_qiskit.py
|   +-- run_pennylane.py
|   +-- run_cudaq.py
|   +-- run_ibm_cloud.py
|   +-- run_ibm_vqe.py
|   +-- run_ibm_vqe_batched.py
|   +-- scale_experiment.py
|   +-- quantum_env_check.py
|   +-- quantum_cloud_check.py
|   +-- quantum_stack_benchmark.py
|   +-- quantum_router.py
|   +-- plot_comparison.py
|   +-- outputs/
+-- src/quant_dev/
+-- docs/
```

## 限制

- 本機 benchmark 數字會受硬體、驅動與系統負載影響。
- naive GPU simulator 是刻意簡化的教育 baseline，不是 production simulator。
- CUDA-Q/cuStateVec 仍然是 classical simulation，不是真實量子硬體。
- IBM Quantum 結果反映 noisy hardware execution，不應直接與 local simulator runtime 比較。
- 固定參數 IBM 結果不是 VQE 收斂結果。
- hardware-in-the-loop VQE 是小型 noisy run，不應解讀為完整收斂的化學計算。

## References

- [Qiskit Learning](https://learning.quantum.ibm.com/)
- [PennyLane Documentation](https://docs.pennylane.ai/)
- [CUDA-Q Documentation](https://docs.nvidia.com/cuda-quantum/latest/)
- [IBM Quantum Platform](https://quantum.ibm.com/)
