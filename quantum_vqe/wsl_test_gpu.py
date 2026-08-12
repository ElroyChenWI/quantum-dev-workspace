"""CUDA-Q GPU 煙霧測試：在 RTX 2060 上取樣 Bell 態。"""
import cudaq

cudaq.set_target("nvidia")
print("目前 target:", cudaq.get_target().name)
print("GPU 可用:", cudaq.get_target().num_qpus(), "個 QPU")


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


result = cudaq.sample(bell, shots_count=1000)
print("Bell 態測量結果:", result)
print("00 機率:", result.count("00") / 1000)
print("11 機率:", result.count("11") / 1000)
print("GPU 煙霧測試成功！")
