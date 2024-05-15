import os
import pickle
from typing import Dict

import gdown
import torch
import yaml
from platformdirs import user_cache_dir

from general_navigation.diffusion_policy.model.diffusion import (
    conditional_unet1d,
)

from .base_model import BaseModel
from .gnm.gnm import GNM
from .nomad.nomad import DenseNetwork, NoMaD
from .nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from .vint.vint import ViNT
from .vint.vit import ViT

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CACHE_DIR = user_cache_dir("general_navigation", "AdityaNG")

os.makedirs(CACHE_DIR, exist_ok=True)

GNM_WEIGHTS = os.path.join(CACHE_DIR, "gnm.pth")
NOMAD_WEIGHTS = os.path.join(CACHE_DIR, "nomad.pth")
VINT_WEIGHTS = os.path.join(CACHE_DIR, "vint.pth")


def get_default_config(name: str = "nomad.yaml") -> Dict:
    yaml_path = os.path.join(CONFIG_DIR, name)

    assert os.path.isfile(yaml_path), f"File not found: {yaml_path}"

    with open(yaml_path, "r") as f:
        default_config = yaml.safe_load(f)

    return default_config


def get_model(config: Dict) -> BaseModel:
    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(  # type: ignore
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(  # type: ignore
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(  # type: ignore
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(  # type: ignore
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(
                f"Vision encoder {config['vision_encoder']} not supported"
            )

        noise_pred_net = conditional_unet1d.ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    return model


class MyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "vint_train" in module:
            module = module.replace("vint_train", "general_navigation")

        return pickle.Unpickler.find_class(self, module, name)


class CustomPickle:
    Unpickler = MyUnpickler


def get_weights(config: Dict, model, device: str) -> torch.nn.Module:
    if config["model_type"] == "gnm":
        weights_id = "1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh"
        weights_path = GNM_WEIGHTS
    elif config["model_type"] == "vint":
        weights_id = "1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5"
        weights_path = VINT_WEIGHTS
    elif config["model_type"] == "nomad":
        weights_id = "1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb"
        weights_path = NOMAD_WEIGHTS
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if not os.path.isfile(weights_path):
        gdown.download(id=weights_id, output=weights_path)

    # checkpoint = torch.load(weights_path, weights_only=True)
    checkpoint = torch.load(
        weights_path, pickle_module=CustomPickle, map_location=device
    )

    if config["model_type"] == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)

    return model
