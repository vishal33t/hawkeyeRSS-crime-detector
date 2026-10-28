
"""
Threaded webcam capture for the crime detection MVP.

The capture thread continuously reads camera frames and retains only
the newest frame. This prevents an inference pipeline from processing
a long backlog of stale frames later.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Read webcam frames in a dedicated background thread."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.frames_captured = 0
        self.last_error: Optional[str] = None

    def start(self) -> "CameraStream":
        """Open the camera and start the capture thread."""
        if self._running:
            return self

        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check camera permissions or try index 1."
            )

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Continuously capture frames without blocking the display."""
        assert self.capture is not None

        while self._running:
            success, frame = self.capture.read()

            if not success:
                self.last_error = "Camera frame read failed."
                time.sleep(0.02)
                continue

            # Keep only the latest frame; don't build a backlog.
            with self._lock:
                self._frame = frame

            self.frames_captured += 1

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return the latest available frame."""
        with self._lock:
            if self._frame is None:
                return False, None

            # Copy so the caller cannot modify the shared frame.
            return True, self._frame.copy()

    def stop(self) -> None:
        """Stop capture, join the thread, and release the camera."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self._thread = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
