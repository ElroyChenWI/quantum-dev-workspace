# Quantum Dev Workspace

跨框架量子開發環境 · Cross-Framework Quantum Development Environment

整合 **Qiskit / PennyLane / Cirq / CUDA-Q** 的量子開發環境，從基礎教學到進階專案，核心為「同一個物理問題、跨三個框架執行」的 VQE 專案。

A quantum development environment integrating **Qiskit / PennyLane / Cirq / CUDA-Q**, from fundamentals to advanced projects. The centerpiece is a VQE project that runs one physics problem across three frameworks.

## 核心展示 / Highlights

- 環境完整度：四框架 + 本機模擬，可擴展至 IBM Quantum 雲端與 CUDA-Q GPU
  Full environment: four frameworks + local simulation, extensible to IBM Quantum cloud and CUDA-Q GPU
- 演算法深度：VQE 找 H₂ 基態能量，收斂到精確值，誤差 < 1e-13 Ha
  Real algorithm: VQE finds the H₂ ground state, converging to the exact value within < 1e-13 Ha
- 工程品質：共用問題定義、多框架對照、結果可驗證
  Engineering: shared problem definition, cross-framework comparison, verifiable results

## 一鍵體驗 / One-Command Demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python demo.py
```

`demo.py` 依序執行 Qiskit / PennyLane VQE，最後畫出兩框架的收斂比較圖，預期兩者都收斂到 **-1.857275 Ha**（誤差 < 1e-13）。

`demo.py` runs the Qiskit and PennyLane VQE in sequence, then plots the convergence comparison. Both converge to **-1.857275 Ha** within < 1e-13.

## VQE 跨框架專案（主角）/ Flagship: One Problem, Three Frameworks

用**同一個 H₂ 分子問題**、**同一個 ansatz**、**同一個最佳化器**，在 Qiskit / PennyLane / CUDA-Q 三個框架實作 VQE，全部收斂到同一個精確基態能量。換框架只是換「量子執行引擎」，問題定義完全共用。

The same H₂ molecule problem, the same ansatz, and the same optimizer, implemented in Qiskit / PennyLane / CUDA-Q. All converge to the exact same ground-state energy — switching framework only swaps the quantum execution engine.

| 框架 / Framework | 基態能量 / Energy (Ha) | 誤差 / Error | 說明 / Notes |
|------|--------------|------|------|
| 精確值（對角化）/ Exact (diagonalization) | -1.857275 | — | 參考基準 / Reference |
| Qiskit | -1.857275 | ~1e-13 | Windows 本機可跑 / Runs on Windows |
| PennyLane | -1.857275 | ~1e-13 | Windows 本機可跑 / Runs on Windows |
| CUDA-Q | 待執行 / TBD | — | 需 WSL2 / Docker，可吃 GPU / WSL2 or Docker; GPU-capable |

![VQE 收斂比較 / VQE convergence comparison](quantum_vqe/outputs/vqe_comparison.png)

> 詳見 / See [quantum_vqe/README.md](quantum_vqe/README.md)

## 使用的框架 / Frameworks

| 框架 / Framework | 用途 / Purpose | 後端 / Backend |
|------|------|------|
| [Qiskit](https://qiskit.org/) | 量子電路、量子演算法 / circuits, quantum algorithms | 本機 Aer 模擬器（可擴展至 IBM Quantum 雲端）/ local Aer simulator (IBM Quantum cloud-ready) |
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
├── quantum_vqe/            # ★ VQE 專案 / flagship VQE project
├── src/quant_dev/          # 可重用程式碼 / reusable code
├── docs/CUDA-Q_SETUP.md    # CUDA-Q 安裝指引 / CUDA-Q setup guide
└── outputs/                # 執行結果圖表 / generated charts
```

## CUDA-Q

CUDA-Q 目前**不支援 Windows 原生執行**，請使用 **WSL2** 或 **Docker**。詳細步驟見 [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md)。

CUDA-Q does **not** run natively on Windows; use **WSL2** or **Docker**. See [docs/CUDA-Q_SETUP.md](docs/CUDA-Q_SETUP.md).

## 之後連上真實量子電腦 / Connect to Real Quantum Hardware

- **IBM Quantum**：到 [IBM Quantum Platform](https://quantum.ibm.com/) 註冊，取得 API token 後設定 `QISKIT_IBM_TOKEN` 環境變數即可在雲端跑 Qiskit 電路。
  Register at [IBM Quantum Platform](https://quantum.ibm.com/), set the `QISKIT_IBM_TOKEN` environment variable, and run Qiskit circuits on the cloud.
- 本專案預設先以本機模擬器開發，程式碼保留接雲端的彈性。
  Developed on local simulators by default, with cloud-ready code.

## 參考資源 / Resources

- [Qiskit 官方教學 / Qiskit Learning](https://learning.quantum.ibm.com/)
- [PennyLane 文件 / PennyLane Docs](https://docs.pennylane.ai/)
- [Cirq 文件 / Cirq Docs](https://quantumai.google/cirq)
- [CUDA-Q 文件 / CUDA-Q Docs](https://docs.nvidia.com/cuda-quantum/latest/)
