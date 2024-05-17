"""
GPT Vision to make Control Decisions
"""

from typing import Dict

import cv2
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from general_navigation.models.factory import (
    get_default_config,
    get_model,
    get_weights,
)
from general_navigation.models.model_utils import model_step
from general_navigation.mpc import MPC
from general_navigation.schema.environment import DroneControls, DroneState


class GPTVision:
    def __init__(
        self,
        config: Dict = get_default_config(),
        device: str = "auto",
    ):
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

        self.mpc = MPC(2.0, 0.035, 6)

    def step(
        self,
        state: DroneState,
    ) -> DroneControls:
        # base64_image = encode_opencv_image(image)
        np_frame = state.image.cv_image()
        frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
        frame = PILImage.fromarray(frame)
        gpt_controls = DroneControls(
            trajectory=[(0, 0), (0, 0)], speed=0.0, steer=0.0
        )

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(frame)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(frame)

        # print("image:", np_frame.shape)
        # print("state:", str(state))

        trajectory = model_step(
            self.model,
            self.noise_scheduler,
            self.context_queue,
            self.config,
            self.device,
        )

        if trajectory is not None:
            gpt_controls.trajectory = trajectory.tolist()
            gpt_controls.speed = 8.33  # 30 mph

            accel, steer = self.mpc.step(trajectory, gpt_controls.speed)

            gpt_controls.steer = steer

        return gpt_controls
