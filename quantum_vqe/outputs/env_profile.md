# Quantum Environment Profile

- Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26220-SP0`
- RAM: `15.835` GB

## Module Availability

| Module | Available | Version / Error |
|---|---:|---|
| `numpy` | True | `2.4.6` |
| `scipy` | True | `1.17.1` |
| `qiskit` | True | `2.5.1` |
| `qiskit_aer` | True | `0.17.2` |
| `pennylane` | True | `0.45.1` |
| `cirq` | True | `1.7.0` |
| `cudaq` | False | `ModuleNotFoundError: No module named 'cudaq'` |
| `cupy` | False | `ModuleNotFoundError: No module named 'cupy'` |

## Hardware

- `nvidia-smi`: True
- NVIDIA GPU: `NVIDIA GeForce RTX 2060, 6144 MiB, 581.42`
- WSL: True

## Smoke Benchmarks

### CPU

| Qubits | Depth | Runtime (s) | <Z0> | Statevector GB |
|---:|---:|---:|---:|---:|
| 4 | 1 | 0.001959 | 0.969012 | 0.000000 |
| 8 | 1 | 0.002820 | 0.908369 | 0.000004 |
| 12 | 1 | 0.004512 | 0.893646 | 0.000061 |

### CUDA-Q GPU
- unavailable: `ModuleNotFoundError: No module named 'cudaq'`

## Recommendations

- CPU simulation available; smoke test suggests <=1s up to n=12.
- CUDA-Q GPU path unavailable here; use WSL2/Linux/Docker with CUDA-Q for GPU runs.
- IBM token is configured; QPU execution can be requested explicitly.
