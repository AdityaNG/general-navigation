"""CLI interface for general_navigation project.
"""

import os

import cv2
import numpy as np
import torch

from .gpt.gpt_vision import GPTVision
from .models.factory import get_default_config
from .models.model_utils import (
    plot_bev_trajectory,
    plot_carstate_frame,
    plot_steering_traj,
)
from .schema.environment import DroneState
from .schema.image import Image


def main(args):  # pragma: no cover
    """
    The main function executes on commands:
    `python -m general_navigation` and `$ general_navigation `.
    """
    device = args.device
    input_media = args.media
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = get_default_config()
    gpt = GPTVision(config, device=device)

    try:
        input_media = int(input_media)
        print(f"Camera index: {input_media}")
    except ValueError:
        assert os.path.isfile(input_media), f"File not found: {input_media}"
        print(f"File path: {input_media}")

    vid = cv2.VideoCapture(input_media)
    ret, np_frame = vid.read()

    steering_angle = 0.0
    iter_count = 0

    with torch.no_grad():
        while ret:
            state = DroneState(
                image=Image(data=np_frame),
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_z=0.0,
                steering_angle=steering_angle,
            )
            controls = gpt.step(state)
            trajectory = np.array(controls.trajectory)
            steering_angle = controls.steer

            np_frame = cv2.resize(np_frame, (256, 128))
            np_frame_bev = np.zeros_like(np_frame)

            trajectory = trajectory * 5.0
            np_frame = plot_steering_traj(
                np_frame,
                trajectory,
                color=(255, 0, 0),
            )
            np_frame_bev = plot_bev_trajectory(
                trajectory, np_frame, color=(255, 0, 0)
            )
            np_frame_bev = plot_carstate_frame(
                np_frame_bev,
                steering_angle,
            )
            print("steering_angle", steering_angle)

            vis_frame = np.hstack((np_frame, np_frame_bev))
            if not args.silent:
                cv2.imshow("General Navigation", vis_frame)

                key = cv2.waitKey(1)
                if ord("q") == key:
                    break

            ret, np_frame = vid.read()
            iter_count += 1

            if iter_count > args.max_iters:
                break
