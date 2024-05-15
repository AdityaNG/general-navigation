"""
Models
"""

from typing import List

import numpy as np
from pydantic import validator

from general_navigation.common import KMPH_2_MPS

from .image import Image
from .packet import Packet


class DroneControls(Packet):
    """Controls Request"""

    trajectory: List[List[float]]  # [(x1, y1),...,(xN, yN)] m
    speed: float  # m/s
    steer: float  # -1 to 1

    @validator("trajectory", pre=True)
    def validate_image(cls, v):
        if isinstance(v, list) or isinstance(v, tuple):
            v = np.array(v, dtype=np.float32)

        # if isinstance(v, dict):
        #     print(v)

        assert len(v.shape) == 2, f"invalid shape: {v.shape}"
        assert v.shape[1] == 2, f"invalid shape: {v.shape}"

        v = v.tolist()

        return v


class DroneState(Packet):
    """Drone State"""

    image: Image  # jpeg encoded image
    velocity_x: float
    velocity_y: float
    velocity_z: float
    steering_angle: float

    def speed_kmph(self) -> float:
        speed_ms = np.sqrt(
            self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2
        )
        speed_kmph = speed_ms / KMPH_2_MPS
        return speed_kmph

    def is_stationary(self) -> bool:
        speed = np.sqrt(
            self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2
        )
        if speed > 5 * KMPH_2_MPS:
            return False

        return True

    def __str__(self) -> str:
        return str(
            f"DroneState(image={np.array(self.image.cv_image()).shape}"
            f"vx={self.velocity_x}, "
            f"vy={self.velocity_y}, "
            f"vz={self.velocity_z}, "
            f"steering={self.steering_angle})"
        )

    # def __repr__(self) -> str:
    #     return str(self)
