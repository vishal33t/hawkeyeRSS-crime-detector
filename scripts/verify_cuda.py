"""
verify_cuda.py  --  Day 1 environment check
===========================================

Purpose
-------
Confirm that PyTorch inside this virtual environment can see and use the
NVIDIA RTX 2050 through CUDA and cuDNN, and that FP16 maths (which runs on
the GPU's Tensor Cores) is actually faster than FP32.

Run it from the project root with the venv activated:

    python scripts/verify_cuda.py

Exit code 0 = everything passed, 1 = something failed.

Why each check exists
---------------------
1. A "CPU-only" PyTorch build is the most common beginner mistake. It installs
   fine and imports fine, but silently runs everything on the CPU.
2. cuDNN is NVIDIA's deep-learning kernel library. Convolutions (the core of
   YOLO) run through it, so it must be available and enabled.
3. Tensor Cores only accelerate FP16/BF16/TF32 maths. If FP16 is not clearly
   faster than FP32 in the benchmark, we are not getting that acceleration.
4. VRAM is only 4 GB, so we measure it now and keep an eye on it all project.
"""

import sys
import time


def section(title: str) -> None:
    """Print a visible section header."""
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def benchmark_matmul(torch, dtype, n: int = 4096, iters: int = 20):
    """
    Time an (n x n) @ (n x n) matrix multiplication on the GPU.

    Returns (milliseconds per multiplication, achieved TFLOPS).

    Notes
    -----
    * CUDA calls are asynchronous: Python returns immediately while the GPU is
      still working. torch.cuda.synchronize() makes the CPU wait, so the timer
      measures real GPU time and not just the time to launch the work.
    * The warm-up loop lets CUDA initialise kernels and caches first, so that
      one-time start-up cost does not distort the result.
    """
    device = torch.device("cuda")
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    for _ in range(3):  # warm-up
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    seconds_each = (time.perf_counter() - start) / iters

    flops = 2 * n ** 3  # a matmul does ~2*n^3 floating-point operations
    tflops = flops / seconds_each / 1e12

    del a, b  # release the tensors so the memory can be reused
    return seconds_each * 1000, tflops


def main() -> int:
    all_ok = True

    # ------------------------------------------------------------------ 1
    section("1. Python and PyTorch")
    print(f"Python version : {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed in this environment.")
        print("       Is the venv activated? Did the pip install finish?")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"Built for CUDA : {torch.version.cuda}")

    if torch.version.cuda is None:
        print("[FAIL] This is a CPU-only PyTorch build.")
        print("       Uninstall it and reinstall using the command generated")
        print("       at https://pytorch.org/get-started/locally/ (choose CUDA).")
        return 1
    print("[ OK ] CUDA-enabled PyTorch build detected.")

    # ------------------------------------------------------------------ 2
    section("2. CUDA availability")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() returned False.")
        print("       Common causes: outdated NVIDIA driver, laptop using the")
        print("       integrated GPU, or a CPU-only build. Run `nvidia-smi`.")
        return 1
    print("[ OK ] CUDA is available.")
    print(f"GPUs visible   : {torch.cuda.device_count()}")

    # ------------------------------------------------------------------ 3
    section("3. GPU properties")
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)

    print(f"Device name        : {props.name}")
    print(f"Compute capability : {major}.{minor}")
    print(f"Total VRAM         : {props.total_memory / 1024 ** 3:.2f} GB")
    print(f"Multiprocessors    : {props.multi_processor_count}")

    # Tensor Cores first appeared with compute capability 7.0 (Volta).
    has_tensor_cores = major >= 7
    print(f"Tensor Cores       : {'yes' if has_tensor_cores else 'NO'}")
    if not has_tensor_cores:
        print("[WARN] This GPU has no Tensor Cores; FP16 will not be faster.")
        all_ok = False

    # ------------------------------------------------------------------ 4
    section("4. cuDNN")
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"cuDNN available : {cudnn_available}")
    print(f"cuDNN enabled   : {torch.backends.cudnn.enabled}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")
    if not (cudnn_available and torch.backends.cudnn.enabled):
        print("[FAIL] cuDNN is not usable.")
        all_ok = False
    else:
        print("[ OK ] cuDNN is ready.")

    # ------------------------------------------------------------------ 5
    section("5. Real GPU computation: FP32 vs FP16 (Tensor Cores)")
    try:
        ms32, tf32 = benchmark_matmul(torch, torch.float32)
        ms16, tf16 = benchmark_matmul(torch, torch.float16)
    except torch.cuda.OutOfMemoryError:
        print("[FAIL] Ran out of GPU memory during the benchmark.")
        print("       Close other GPU-heavy apps (games, browsers with GPU")
        print("       acceleration, video editors) and try again.")
        return 1

    print(f"FP32 : {ms32:8.2f} ms per matmul  ({tf32:6.2f} TFLOPS)")
    print(f"FP16 : {ms16:8.2f} ms per matmul  ({tf16:6.2f} TFLOPS)")
    speedup = ms32 / ms16
    print(f"FP16 speed-up over FP32: {speedup:.2f}x")

    if speedup >= 1.5:
        print("[ OK ] FP16 is clearly faster, so Tensor Cores are being used.")
    else:
        print("[WARN] FP16 speed-up is small. Plug in the charger, set Windows")
        print("       to 'Best performance', and close other apps, then re-run.")
        all_ok = False

    # ------------------------------------------------------------------ 6
    section("6. GPU memory handling")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"Free / total VRAM (driver view): "
          f"{free_b / 1024 ** 3:.2f} / {total_b / 1024 ** 3:.2f} GB")
    print(f"Allocated by PyTorch tensors   : "
          f"{torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"Reserved by PyTorch cache      : "
          f"{torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    # PyTorch keeps freed memory in a cache for fast reuse. empty_cache()
    # hands it back to the driver. We will rarely need this in the real
    # pipeline, but it is useful to know the difference between
    # "allocated" (in use) and "reserved" (cached) when debugging VRAM.
    torch.cuda.empty_cache()
    print(f"Reserved after empty_cache()   : "
          f"{torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    # ------------------------------------------------------------------ end
    section("RESULT")
    if all_ok:
        print("[ OK ] Your RTX GPU is ready for the project.")
        return 0
    print("[WARN] Some checks failed or warned. Read the messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
