"""
Models
"""

import time

from pydantic import BaseModel, Field


class Packet(BaseModel):
    """Packet"""

    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
