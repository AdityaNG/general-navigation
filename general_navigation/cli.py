"""CLI interface for general_navigation project.
"""

import os
import time

import cv2
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from .models.factory import get_default_config, get_model, get_weights
from .models.model_utils import (
    get_action,
    plot_bev_trajectory,
    plot_steering_traj,
    to_numpy,
    transform_images,
)

MAX_V = 1.0
MAX_W = 1.0
RATE = 10.0


def model_step(
    model,
    noise_scheduler,
    context_queue,
    model_params,
    device,
    num_samples=8,
    waypoint=2,
):
    if len(context_queue) > model_params["context_size"]:

        obs_images = transform_images(
            context_queue, model_params["image_size"], center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(device)
        fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
        mask = torch.ones(1).long().to(device)  # ignore the goal

        # infer action
        # encoder vision features
        obs_cond = model(
            "vision_encoder",
            obs_img=obs_images,
            goal_img=fake_goal,
            input_goal_mask=mask,
        )

        # (B, obs_horizon * obs_dim)
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(num_samples, 1, 1)

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (num_samples, model_params["len_traj_pred"], 2), device=device
        )
        naction = noisy_action

        if noise_scheduler is not None:
            # init scheduler
            noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])

        start_time = time.time()
        for k in noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )

            if noise_scheduler is not None:
                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
            else:
                naction = noise_pred
        print("time elapsed:", time.time() - start_time)

        naction = to_numpy(get_action(naction))

        naction = naction[0]  # change this based on heuristic

        chosen_waypoint = naction[waypoint]

        if model_params["normalize"]:
            chosen_waypoint *= MAX_V / RATE

        return naction

    return None


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
