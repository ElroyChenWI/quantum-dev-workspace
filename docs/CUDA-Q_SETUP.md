# CUDA-Q 安裝與使用指引

[CUDA-Q](https://developer.nvidia.com/cuda-q) 是 NVIDIA 的量子開發平台，主打**高效能 GPU 量子模擬**與**混合量子-經典運算**。

> **重要**：CUDA-Q 目前**不支援 Windows 原生執行**。請使用 **WSL2** 或 **Docker**。

---

## 方法一：WSL2（建議，若你有 NVIDIA GPU）

### 1. 安裝 WSL2
在 **Windows PowerShell（系統管理員）** 執行：

```powershell
wsl --install -d Ubuntu-22.04
```

重新啟動電腦後，確認版本為 2：

```powershell
wsl -l -v
```

### 2. 在 WSL 內安裝 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 3. 建立環境並安裝 CUDA-Q

```bash
conda create -n cudaq python=3.10
conda activate cudaq
conda install -c nvidia cuda-quantum
```

> 若無 GPU 也可安裝 CPU-only 版：`pip install cuda-quantum`

### 4. 驗證安裝

```bash
python -c "import cudaq; print(cudaq.__version__)"
```

---

## 方法二：Docker（不依賴本機 Python 環境）

```bash
docker pull nvcr.io/nvidia/quantum/cuda-quantum:latest
docker run -it --gpus all nvcr.io/nvidia/quantum/cuda-quantum:latest
```

---

## 快速範例

```python
import cudaq

# 定義量子 kernel（NVIDIA 自訂語法）
@cudaq.kernel
def bell_pair():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)

# 本機 GPU 模擬
result = cudaq.sample(bell_pair)
print(result)
```

### 與其他框架整合

- **Qiskit**：CUDA-Q 可讀取 Qiskit 電路（透過 QIR 介面），從 `qiskit` 匯入電路到 `cudaq` 執行。
- **PennyLane**：安裝 `pennylane-cudaq` 外掛後，可用 `qml.device("cudaq")` 將 PennyLane 電路跑在 GPU 上。

---

## 常見問題

| 問題 | 解法 |
|------|------|
| `wsl --install` 失敗 | 確認已啟用「Windows 子系統 Linux 版」功能，並重啟電腦 |
| GPU 偵測不到 | 在 WSL 內執行 `nvidia-smi`，若無顯示請更新 NVIDIA 驅動（須支援 WSL） |
| 安裝很慢 | 可改用 Docker 方法，或設定 conda 鏡像 |

---

## 之後上雲端

CUDA-Q 也支援連接到遠端量子硬體與 HPC 叢集。開發階段先用本機模擬器即可，需要時再參考 [官方文件](https://docs.nvidia.com/cuda-quantum/latest/)。
