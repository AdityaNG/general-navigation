"""
GPT Vision to make Control Decisions
"""

from typing import Dict

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from general_navigation.models.factory import (
    get_default_config,
    get_model,
    get_weights,
)
from general_navigation.models.model_utils import model_step
from general_navigation.schema.carla import DroneControls, DroneState


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

    def step(
        self,
        state: DroneState,
    ) -> DroneControls:
        # base64_image = encode_opencv_image(image)
        image = state.image.cv_image()
        gpt_controls = DroneControls(trajectory=[(0, 0), (0, 0)], speed=0.0)

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(image)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(image)

        print("image:", image.shape)
        print("state:", state)

        trajectory = model_step(
            self.model,
            self.noise_scheduler,
            self.context_queue,
            self.config,
            self.device,
        )

        if trajectory is not None:
            gpt_controls.trajectory = gpt_controls
            gpt_controls.speed = 1.0

        print("gpt:", gpt_controls)

        return gpt_controls
