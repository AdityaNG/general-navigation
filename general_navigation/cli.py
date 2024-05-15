"""CLI interface for general_navigation project.
"""

import os

import cv2
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from .models.factory import get_default_config, get_model, get_weights
from .models.model_utils import (
    model_step,
    plot_bev_trajectory,
    plot_steering_traj,
)


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
    model = get_model(config)
    model = get_weights(config, model, device)

    noise_scheduler = None
    if config["run_name"] == "nomad":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    model = model.to(device=device)

    context_queue = []

    try:
        input_media = int(input_media)
        print(f"Camera index: {input_media}")
    except ValueError:
        assert os.path.isfile(input_media), f"File not found: {input_media}"
        print(f"File path: {input_media}")

    vid = cv2.VideoCapture(input_media)
    ret, frame = vid.read()

    context_size = config["context_size"]

    with torch.no_grad():
        while ret:
            trajectory = model_step(
                model,
                noise_scheduler,
                context_queue,
                config,
                device,
            )
            ret, np_frame = vid.read()
            frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            frame = PILImage.fromarray(frame)

            if len(context_queue) < context_size + 1:
                context_queue.append(frame)
            else:
                context_queue.pop(0)
                context_queue.append(frame)

            np_frame = cv2.resize(np_frame, (256, 128))
            np_frame_bev = np.zeros_like(np_frame)
            if trajectory is not None:
                trajectory = trajectory * 5.0
                np_frame = plot_steering_traj(
                    np_frame,
                    trajectory,
                    color=(255, 0, 0),
                )
                np_frame_bev = plot_bev_trajectory(
                    trajectory, np_frame, color=(255, 0, 0)
                )

            vis_frame = np.hstack((np_frame, np_frame_bev))
            cv2.imshow("General Navigation", vis_frame)

            key = cv2.waitKey(1)
            if ord("q") == key:
                break
