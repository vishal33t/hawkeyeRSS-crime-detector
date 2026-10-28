
"""Test webcam capture, display, resolution, and FPS."""

import sys
import time
from pathlib import Path

import cv2

# Add the project root so this script works when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraStream


def main() -> None:
    camera = CameraStream(camera_index=0)
    display_frames = 0
    start_time = time.perf_counter()
    fps = 0.0
    last_fps_update = start_time

    try:
        camera.start()
        print("Camera started. Press 'q' to quit.")

        while True:
            available, frame = camera.read()

            if not available or frame is None:
                time.sleep(0.005)
                continue

            display_frames += 1
            now = time.perf_counter()
            elapsed = now - last_fps_update

            if elapsed >= 1.0:
                fps = display_frames / (now - start_time)
                last_fps_update = now

            height, width = frame.shape[:2]

            cv2.putText(
                frame,
                f"Display FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Resolution: {width}x{height}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Crime Detection - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as exc:
        print(f"Camera error: {exc}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"Frames captured: {camera.frames_captured}")
        print(f"Last camera error: {camera.last_error}")
        print("Camera resources released.")


if __name__ == "__main__":
    main()
