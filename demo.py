"""
一鍵體驗：跑完 VQE 跨框架專案的完整流程
=====================================

依序執行：
    1. run_qiskit.py     Qiskit VQE（H2 基態能量）
    2. run_pennylane.py  PennyLane VQE（H2 基態能量）
    3. plot_comparison.py  畫出兩框架的收斂比較圖

使用方式（先啟動虛擬環境）：
    .\.venv\Scripts\Activate.ps1
    python demo.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
VQE = ROOT / "quantum_vqe"


def run(script: Path) -> None:
    print(f"\n{'=' * 60}\n>>> 執行 {script.name}\n{'=' * 60}")
    subprocess.run([PYTHON, str(script)], cwd=VQE, check=True)


def main() -> None:
    print("Quantum Dev Workspace — 一鍵體驗")
    run(VQE / "run_qiskit.py")
    run(VQE / "run_pennylane.py")
    run(VQE / "plot_comparison.py")

    chart = VQE / "outputs" / "vqe_comparison.png"
    print(f"\n完成！收斂比較圖已產生：{chart}")
    print("預期：Qiskit 與 PennyLane 都收斂到 -1.857275 Ha（誤差 < 1e-13）")


if __name__ == "__main__":
    main()
