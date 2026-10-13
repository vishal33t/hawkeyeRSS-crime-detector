# Real-Time Crime and Anomaly Detection System

A GPU-accelerated video surveillance MVP that watches a webcam (or video file)
and raises Telegram alerts for five kinds of events:

1. Weapon brandishing
2. Physical violence / fighting
3. Property damage / vandalism
4. Theft / snatching
5. Loitering / trespassing

Long-term goal: scale the design to many street cameras with an edge-cloud
architecture.

## Tech stack (all free / open-source)
Python 3.10+, PyTorch (CUDA), OpenCV, Ultralytics YOLO (detection, pose,
tracking), Telegram Bot API.

## Architecture (planned)
Camera thread -> latest-frame queue -> GPU inference thread
-> event rules engine -> alert thread (Telegram). The main thread only
displays video, so the UI never freezes.

## Project structure
```
src/        application code (camera, detector, rules, alerts, main)
scripts/    one-off utilities (setup, GPU verification)
tests/      automated tests
models/     model weights (git-ignored)
data/       test videos (git-ignored)
logs/       event logs and snapshots (git-ignored)
docs/       notes, dev log, diagrams
```

## Setup (Windows 10, PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install PyTorch using the command generated at https://pytorch.org/get-started/locally/
python scripts\verify_cuda.py
```
Or run `scripts\setup_day1.ps1` (see the script header for usage).

## Progress
| Day | Topic | Status |
|-----|-------|--------|
| 1 | Workspace, venv, Git, CUDA verification | in progress |
