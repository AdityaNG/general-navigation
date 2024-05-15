import base64

import cv2
import numpy as np

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
