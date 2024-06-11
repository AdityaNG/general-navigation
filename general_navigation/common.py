import base64
import os
import time

import cv2
import numpy as np

from general_navigation.settings import settings

KMPH_2_MPS = 1 / 3.6
DEG_2_RAD = np.pi / 180

MAX_STEER = 460.0
WHEEL_BASE = 2.83972
STEERING_RATIO = 13.27


def encode_opencv_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def encode_opencv_image_buf(img, compression_factor=95):
    _, buffer = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_factor]
    )
    return buffer


def decode_opencv_image_buf(buf):
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class UIRecorder:

    def __init__(self) -> None:
        now = time.strftime("%Y-%m-%d--%H-%M-%S")
        self.path = os.path.join("test_media", f"general-navigation-{now}.mp4")
        self.enabled = settings.ui.RECORDING_ENABLED
        self.fps = 30
        self.cap = None

    def write(self, image) -> None:
        if self.cap is None:
            size = image.shape[:2]
            self.cap = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size
            )

        self.cap.write(image)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
