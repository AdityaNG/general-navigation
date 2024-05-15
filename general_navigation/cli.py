"""CLI interface for general_navigation project.
"""

import time

import cv2
import PIL
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from .models.factory import get_default_config, get_model, get_weights
from .models.model_utils import get_action, to_numpy, transform_images
from .visualizing.action_utils import plot_trajs_and_points_on_image

MAX_V = 1.0
MAX_W = 1.0
RATE = 10.0


def loop(
    model,
    noise_scheduler,
    context_queue,
    model_params,
    device,
    num_samples=8,
    waypoint=2,
):
    if len(context_queue) > model_params["context_size"]:

        num_diffusion_iters = model_params["num_diffusion_iters"]
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

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        start_time = time.time()
        for k in noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        print("time elapsed:", time.time() - start_time)

        naction = to_numpy(get_action(naction))

        naction = naction[0]  # change this based on heuristic

        chosen_waypoint = naction[waypoint]

        if model_params["normalize"]:
            chosen_waypoint *= MAX_V / RATE

        print(chosen_waypoint)


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m general_navigation` and `$ general_navigation `.
    """
    device = "cpu"
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = get_default_config()
    model = get_model(config)
    model = get_weights(config, model, device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    model = model.to(device=device)

    context_queue = []

    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    context_size = config["context_size"]

    with torch.no_grad():
        while ret:
            loop(
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


            visualize_traj_pred(
                to_numpy(obs_image),
                to_numpy(goal_image),
                to_numpy(dataset_index),
                to_numpy(goal_pos),
                to_numpy(action_pred),
                to_numpy(action_label),
                mode,
                normalized,
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
            cv2.imshow("General Navigation", np_frame)

            key = cv2.waitKey(1)
            if ord("q") == key:
                break
