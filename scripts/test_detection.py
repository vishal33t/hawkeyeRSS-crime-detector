
"""Day 3: Real-time YOLOv8 detection using the NVIDIA GPU."""

import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraStream


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch GPU setup first."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading YOLOv8 nano model...")

    # The pretrained COCO model will download automatically
    # the first time it is used.
    model = YOLO("yolov8n.pt")

    camera = CameraStream(camera_index=0)
    inference_times = []
    display_frames = 0
    start_time = time.perf_counter()

    try:
        camera.start()
        print("Detection started. Press 'q' to quit.")

        while True:
            available, frame = camera.read()

            if not available or frame is None:
                time.sleep(0.005)
                continue

            inference_start = time.perf_counter()

            result = model.predict(
                source=frame,
                imgsz=640,
                conf=0.35,
                device=0,
                verbose=False,
            )[0]

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            inference_times.append(inference_ms)
            if len(inference_times) > 30:
                inference_times.pop(0)

            average_inference_ms = sum(inference_times) / len(
                inference_times
            )

            # Draw bounding boxes and detected class names.
            annotated_frame = result.plot()

            display_frames += 1
            elapsed = time.perf_counter() - start_time
            display_fps = display_frames / elapsed

            cv2.putText(
                annotated_frame,
                f"Display FPS: {display_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Inference: {average_inference_ms:.1f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Crime Detection - YOLOv8 GPU Test", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"Camera frames captured: {camera.frames_captured}")

        if inference_times:
            print(
                "Average inference time (last 30 frames): "
                f"{sum(inference_times) / len(inference_times):.2f} ms"
            )

        print("Camera and display resources released.")


if __name__ == "__main__":
    main()
