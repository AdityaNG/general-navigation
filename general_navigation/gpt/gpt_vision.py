"""
GPT Vision to make Control Decisions
"""

from typing import Dict

import cv2
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from general_navigation.models.factory import (
    get_default_config,
    get_model,
    get_weights,
)
from general_navigation.models.model_utils import (
    get_action_torch,
    transform_images,
)
from general_navigation.mpc import MPC
from general_navigation.schema.environment import DroneControls, DroneState


class GPTVision(torch.nn.Module):
    def __init__(
        self,
        config: Dict = get_default_config(),
        device: str = "auto",
    ):
        super(GPTVision, self).__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.config = config
        self.model = get_model(self.config)
        self.model = get_weights(self.config, self.model, self.device)

        if config["run_name"] == "nomad":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        self.model = self.model.to(device=self.device)

        self.context_queue = []
        self.context_size = config["context_size"]

        self.speed = 5.0  # m/s
        self.time_step = 0.05
        self.horizon = 25

        self.mpc = MPC(self.speed, self.time_step, self.horizon)

        self.trajectory_history = []
        self.trajectory_history_size = 2
        self.num_samples = 8

    def append_context(
        self,
        state: DroneState,
    ):
        # base64_image = encode_opencv_image(image)
        np_frame = state.image.cv_image()
        frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
        frame = PILImage.fromarray(frame)

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(frame)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(frame)

    def context_to_torch(
        self,
    ) -> torch.Tensor:
        obs_images = transform_images(
            self.context_queue, self.config["image_size"], center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(self.device)
        return obs_images

    def step(
        self,
        state: DroneState,
    ) -> DroneControls:
        # base64_image = encode_opencv_image(image)
        self.append_context(state)

        gpt_controls = DroneControls(
            trajectory=[(0, 0), (0, 0)],
            trajectory_mpc=[(0, 0), (0, 0)],
            speed=0.0,
            steer=0.0,
        )

        if len(self.context_queue) <= self.config["context_size"]:
            return gpt_controls

        with torch.no_grad():
            obs_images = self.context_to_torch()
            trajectory = self.forward(obs_images)

        trajectory = trajectory.cpu().detach().numpy()

        trajectory = trajectory[0]

        if trajectory is not None:
            self.trajectory_history.append(trajectory.tolist())

            if len(self.trajectory_history) > self.trajectory_history_size:
                self.trajectory_history.pop(0)

            gpt_controls.trajectory = np.mean(
                np.array(self.trajectory_history), axis=0
            )
            gpt_controls.trajectory = gpt_controls.trajectory.tolist()
            gpt_controls.speed = self.speed

            accel, steer, traj_mpc = self.mpc.step(
                np.array(gpt_controls.trajectory),
                state.speed_ms(),
                state.steering_angle,
            )

            gpt_controls.steer = steer
            gpt_controls.trajectory_mpc = traj_mpc

        return gpt_controls

    def forward(self, obs_images: torch.Tensor) -> torch.Tensor:
        fake_goal = torch.randn((1, 3, *self.config["image_size"])).to(
            self.device
        )
        mask = torch.ones(1).long().to(self.device)  # ignore the goal

        # infer action
        # encoder vision features
        obs_cond = self.model(
            "vision_encoder",
            obs_img=obs_images,
            goal_img=fake_goal,
            input_goal_mask=mask,
        )

        # (B, obs_horizon * obs_dim)
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(self.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (self.num_samples, self.config["len_traj_pred"], 2),
            device=self.device,
        )
        naction = noisy_action

        if self.noise_scheduler is not None:
            # init scheduler
            self.noise_scheduler.set_timesteps(
                self.config["num_diffusion_iters"]
            )

        for k in self.noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = self.model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )

            if self.noise_scheduler is not None:
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
            else:
                naction = noise_pred

        trajectory = get_action_torch(naction)

        # if self.config["normalize"]:
        #     trajectory *= 1.0 / 10.0

        trajectory[:, :, [0, 1]] = trajectory[:, :, [1, 0]]

        return trajectory
