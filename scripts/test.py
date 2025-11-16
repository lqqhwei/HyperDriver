import torch, cupy as cp

print("=== PyTorch ===")
print("cuda available:", torch.cuda.is_available(), " | torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

print("\n=== CuPy ===")
print("cupy version:", cp.__version__)
try:
    rt_ver = cp.cuda.runtime.runtimeGetVersion()
    drv_ver = cp.cuda.runtime.driverGetVersion()
    ndev    = cp.cuda.runtime.getDeviceCount()
    print("cuda runtime:", rt_ver, " | driver:", drv_ver, " | device count:", ndev)
    # 简单算一把确认 GPU 在干活
    x = cp.arange(10**6, dtype=cp.float32)
    print("gpu sum:", float(x.sum()))
except Exception as e:
    print("CuPy runtime check failed:", type(e).__name__, e)

print("\n=== CuPy config ===")
cp.show_config()
