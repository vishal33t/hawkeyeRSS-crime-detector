
"""Day 4: Measure camera, YOLO inference and processing performance."""

import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraStream


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")

    print("Loading YOLOv8 nano...")
    model = YOLO("yolov8n.pt")

    camera = CameraStream(camera_index=0)

    # Performance measurement variables
    inference_times = []
    processing_times = []

    frame_count = 0
    last_report_time = time.perf_counter()
    report_frames = 0
    report_processing_times = []
    report_inference_times = []

    try:
        camera.start()
        print("\nPerformance test started. Press 'q' to quit.")
        print("Allow the application to run for at least 60 seconds.\n")

        while True:
            loop_start = time.perf_counter()

            available, frame = camera.read()

            if not available or frame is None:
                time.sleep(0.005)
                continue

            # Synchronize CUDA before measuring inference.
            torch.cuda.synchronize()
            inference_start = time.perf_counter()

            result = model.predict(
                source=frame,
                imgsz=640,
                conf=0.35,
                device=0,
                verbose=False,
            )[0]

            torch.cuda.synchronize()
            inference_end = time.perf_counter()

            inference_ms = (inference_end - inference_start) * 1000

            inference_times.append(inference_ms)
            report_inference_times.append(inference_ms)

            # Draw detection boxes.
            annotated_frame = result.plot()

            frame_count += 1
            report_frames += 1

            processing_ms = (
                time.perf_counter() - loop_start
            ) * 1000

            processing_times.append(processing_ms)
            report_processing_times.append(processing_ms)

            # Keep only recent samples for display.
            inference_times = inference_times[-30:]
            processing_times = processing_times[-30:]

            avg_inference = sum(inference_times) / len(inference_times)
            avg_processing = sum(processing_times) / len(processing_times)

            now = time.perf_counter()
            elapsed = now - last_report_time

            if elapsed >= 1.0:
                processing_fps = report_frames / elapsed
                camera_fps = camera.frames_captured / (
                    now - start_time
                )

                avg_inf = sum(report_inference_times) / len(
                    report_inference_times
                )

                avg_proc = sum(report_processing_times) / len(
                    report_processing_times
                )

                print(
                    f"Camera: {camera_fps:.1f} FPS | "
                    f"Processing: {processing_fps:.1f} FPS | "
                    f"Inference: {avg_inf:.1f} ms | "
                    f"Processing time: {avg_proc:.1f} ms"
                )

                last_report_time = now
                report_frames = 0
                report_inference_times.clear()
                report_processing_times.clear()

            cv2.putText(
                annotated_frame,
                f"Processing FPS: {1 / (avg_processing / 1000):.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Inference: {avg_inference:.1f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Crime Detection - Day 4 Performance", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        print("\n========== PERFORMANCE SUMMARY ==========")
        print(f"Total processed frames: {frame_count}")
        print(f"Camera frames captured: {camera.frames_captured}")

        if inference_times:
            print(
                "Recent average inference time: "
                f"{sum(inference_times) / len(inference_times):.2f} ms"
            )

        print(
            "Peak GPU memory allocated: "
            f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB"
        )

        print(
            "Current GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB"
        )

        print("Performance test finished.")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
