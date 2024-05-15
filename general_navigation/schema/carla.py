"""
Models
"""

import numpy as np
from pydantic import ConfigDict

from general_navigation.common import KMPH_2_MPS

from .image import Image
from .packet import Packet


class DroneControls(Packet):
    """Controls Request"""

    trajectory: np.ndarray  # m
    speed: float  # m/s

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
