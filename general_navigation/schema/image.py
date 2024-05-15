"""
Models
"""

import numpy as np
from pydantic import validator

from general_navigation.common import (
    decode_opencv_image_buf,
    encode_opencv_image_buf,
)

from .packet import Packet


class Image(Packet):
    """Image which is encoded as a JPEG"""

    data: tuple  # jpeg encoded image

    def cv_image(self):
        buffer = np.array(self.data, dtype=np.uint8)
        img_decoded = decode_opencv_image_buf(buffer)
        img = np.array(img_decoded, dtype=np.uint8)
        return img

    @validator("data", pre=True)
    def validate_image(cls, v):
        if isinstance(v, list) or isinstance(v, tuple):
            v = np.array(v, dtype=np.uint8)

        if len(v.shape) == 1:
            return v.tolist()
        v = encode_opencv_image_buf(v)

        assert len(v.shape) == 1
        v = v.tolist()
        return v
