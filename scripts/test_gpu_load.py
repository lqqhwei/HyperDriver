# test_gpu_load.py
import torch, time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda?", torch.cuda.is_available(), "| device:", device, "| name:", torch.cuda.get_device_name(0))

# 连续跑 10 秒 GEMM，任务管理器改到 GPU1 的 CUDA/Compute 曲线看占用
A = torch.randn(8192, 8192, device=device)
B = torch.randn(8192, 8192, device=device)
t0 = time.time()
iters = 0
while time.time() - t0 < 10:
    C = A @ B  # 大矩阵乘法，强占用
    iters += 1
torch.cuda.synchronize()
print("iters in 10s:", iters)
