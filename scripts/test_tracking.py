
"""Day 5: Real-time object tracking using YOLOv8 and ByteTrack."""

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
        raise RuntimeError("CUDA is unavailable.")

    print("GPU:", torch.cuda.get_device_name(0))

    print("Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")

    print("Model device:", next(model.model.parameters()).device)

    camera = CameraStream(camera_index=0)

    processed_frames = 0
    unique_ids_seen = set()
    start_time = time.perf_counter()

    try:
        camera.start()
        print("\nByteTrack started. Press 'q' to quit.")

        while True:
            available, frame = camera.read()

            if not available or frame is None:
                time.sleep(0.005)
                continue

            # Track objects and retain their IDs between frames.
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=640,
                conf=0.35,
                device=0,
                verbose=False,
            )

            result = results[0]
            print(
                  "\nInference device:",
                    model.predictor.device
             )
            # Draw bounding boxes, labels and tracking IDs.
            annotated_frame = result.plot()

            tracked_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                tracked_count = len(track_ids)
                unique_ids_seen.update(track_ids)

                for track_id, class_id in zip(track_ids, class_ids):
                    class_name = model.names[class_id]
                    print(
                        f"\rTracking: ID {track_id} | "
                        f"Object: {class_name} | "
                        f"Currently tracked: {tracked_count}",
                        end="",
                        flush=True,
                    )

            processed_frames += 1

            elapsed = time.perf_counter() - start_time
            processing_fps = processed_frames / elapsed

            cv2.putText(
                annotated_frame,
                f"Processing FPS: {processing_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Tracked objects: {tracked_count}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Crime Detection - YOLOv8 ByteTrack",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        print("\n\n========== TRACKING SUMMARY ==========")
        print("Processed frames:", processed_frames)
        print("Different tracking IDs observed:", len(unique_ids_seen))
        print("IDs observed:", sorted(unique_ids_seen))
        print("Camera frames captured:", camera.frames_captured)
        print("Tracking test finished.")


if __name__ == "__main__":
    main()
