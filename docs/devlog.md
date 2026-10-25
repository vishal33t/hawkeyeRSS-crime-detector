# Dev Log

## Day 1: Workspace, Git and GPU verification

**What I did**
- Created the project structure (src, scripts, tests, docs, models, data, logs)
- Created a virtual environment and installed CUDA-enabled PyTorch
- Ran scripts/verify_cuda.py to confirm the RTX 2050 works with CUDA/cuDNN
- Set up Git with a .gitignore and made the first commit

# Day 1.5— Environment and GPU Verification

- Created and activated the project virtual environment.
- Verified CUDA-enabled PyTorch and cuDNN on the NVIDIA RTX 2050 (4 GB VRAM).
- Confirmed real GPU tensor computation and benchmarked FP32 versus FP16.
- Next: verify OpenCV and Ultralytics, finalize Git setup, and test webcam capture.

**Learning:** CUDA availability means PyTorch can access the GPU. FP16 can accelerate suitable workloads, but actual performance depends on the model and workload.
**What confused me:**

**What I learned:**
