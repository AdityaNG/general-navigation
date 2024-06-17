import os
import textwrap
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import wandb
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from torchvision import transforms

from general_navigation.schema.environment import DroneControls, DroneState
from general_navigation.visualizing.action_utils import plot_trajs_and_points
from general_navigation.visualizing.visualize_utils import from_numpy, to_numpy

MAX_V = 1.0
MAX_W = 1.0
RATE = 10.0


def print_text_image(
    img,
    text,
    width=50,
    font_size=0.5,
    font_thickness=2,
    text_color=(255, 255, 255),
    text_color_bg=(0, 0, 0),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    wrapped_text = textwrap.wrap(text, width=width)

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        text_w, text_h = textsize

        gap = textsize[1] + 10

        y = (i + 1) * gap
        x = 10

        cv2.rectangle(
            img, (x, y - text_h), (x + text_w, y + text_h), text_color_bg, -1
        )
        cv2.putText(
            img,
            line,
            (x, y),
            font,
            font_size,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def generate_visual(
    drone_state: DroneState, gpt_controls: DroneControls
) -> np.ndarray:
    image_raw = np.array(
        drone_state.image.cv_image(),
    )

    # # Draw all template trajectories
    # for index in range(NUM_TEMLATES):
    #     template_trajectory_3d = select_trajectory_index(
    #         trajectory_templates, index
    #     )

    #     color = colors[index]
    #     plot_steering_traj(
    #         image, template_trajectory_3d, color=color, track=False
    #     )

    # print_text_image(visual[0:128, 0:256], "Prompt")

    image_vis = image_raw.copy()

    trajectory = np.array(gpt_controls.trajectory)
    plot_steering_traj(
        image_vis,
        trajectory,
        color=(255, 0, 0),
        track=True,
    )
    trajectory_mpc = np.array(gpt_controls.trajectory_mpc)
    plot_steering_traj(
        image_vis,
        trajectory_mpc,
        color=(0, 255, 0),
        track=True,
    )
    image_bev = plot_bev_trajectory(trajectory, image_vis, color=(255, 0, 0))
    image_bev_mpc = plot_bev_trajectory(
        trajectory_mpc, image_vis, color=(0, 255, 0)
    )
    image_bev = cv2.addWeighted(image_bev, 0.5, image_bev_mpc, 0.5, 0.0)

    visual = np.hstack((image_vis, image_bev))
    print_text_image(visual, "GNM Controls")

    return visual


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

        naction = to_numpy(get_action(naction))

        naction = naction[0]  # change this based on heuristic

        chosen_waypoint = naction[waypoint]

        if model_params["normalize"]:
            chosen_waypoint *= MAX_V / RATE

        trajectory = naction.copy()

        trajectory[:, [0, 1]] = trajectory[:, [1, 0]]
        # trajectory[:, 0] = -trajectory[:, 0]

        return trajectory

    return None


VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3

# LOAD DATA CONFIG
with open(
    os.path.join(os.path.dirname(__file__), "config/data_config.yaml"), "r"
) as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config["action_stats"]:
    ACTION_STATS[key] = np.array(data_config["action_stats"][key])


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def unnormalize_data_torch(ndata: torch.tensor, stats):
    device = ndata.device
    ndata = (ndata + 1) / 2
    data = ndata * torch.tensor(
        stats["max"] - stats["min"], device=device
    ) + torch.tensor(stats["min"], device=device)
    return data


def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action_torch(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = unnormalize_data_torch(ndeltas, action_stats)
    actions = torch.cumsum(ndeltas, dim=1)
    return actions


def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device
    )
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1)
            .repeat(diffusion_output.shape[0])
            .to(device),
            global_cond=obs_cond,
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device
    )
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1)
            .repeat(diffusion_output.shape[0])
            .to(device),
            global_cond=obsgoal_cond,
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = False,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(
        num_images_log,
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_action_label.shape[0],
        batch_goal_pos.shape[0],
    )
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(
        batch_obs_images, max_batch_size, dim=0
    )
    batch_goal_images_list = torch.split(
        batch_goal_images, max_batch_size, dim=0
    )

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict["uc_actions"]))
        gc_actions_list.append(to_numpy(model_output_dict["gc_actions"]))
        gc_distances_list.append(to_numpy(model_output_dict["gc_distance"]))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate(
            [
                uc_actions,
                gc_actions,
                action_label[None],
            ],
            axis=0,
        )
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = (
            ["red"] * len(uc_actions)
            + ["green"] * len(gc_actions)
            + ["magenta"]
        )
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
        )

        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title("diffusion action predictions")
        ax[1].set_title("observation")
        ax[2].set_title(
            f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}"  # noqa
        )

        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)


def transform_images(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    if not isinstance(pil_imgs, list):
        pil_imgs = [
            pil_imgs,  # type: ignore
        ]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(
                    pil_img, (h, int(h * IMAGE_ASPECT_RATIO))
                )  # crop to the right ratio
            else:
                pil_img = TF.center_crop(
                    pil_img, (int(w / IMAGE_ASPECT_RATIO), w)
                )
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def interpolate_trajectory(
    trajectory: np.ndarray,
    samples: int = 1,
) -> np.ndarray:
    """
    Interpolates the trajectory (N, 2) to (M, 2)
    Where M = N*(S+1)+1

    :param trajectory: (N,2) numpy trajectory
    :type trajectory: np.ndarray
    :param samples: number of samples
    :type samples: int
    :returns: (M,2) interpolated numpy trajectory
    """
    # Calculate the number of segments
    num_segments = trajectory.shape[0] - 1

    # Generate the interpolated trajectory
    interpolated_trajectory = np.zeros((num_segments * (samples + 1) + 1, 2))

    # Fill in the interpolated points
    for i in range(num_segments):
        start = trajectory[i]
        end = trajectory[i + 1]
        interpolated_trajectory[
            i * (samples + 1) : (i + 1) * (samples + 1)
        ] = np.linspace(start, end, samples + 2)[:-1]

    # Add the last point
    interpolated_trajectory[-1] = trajectory[-1]

    return interpolated_trajectory


def estimate_intrinsics(
    fov_x: float,  # degrees
    fov_y: float,  # degrees
    height: int,  # pixels
    width: int,  # pixels
) -> np.ndarray:
    """
    The intrinsic matrix can be extimated from the FOV and image dimensions

    :param fov_x: FOV on x axis in degrees
    :type fov_x: float
    :param fov_y: FOV on y axis in degrees
    :type fov_y: float
    :param height: Height in pixels
    :type height: int
    :param width: Width in pixels
    :type width: int
    :returns: (3,3) intrinsic matrix
    """
    c_x = width / 2.0
    c_y = height / 2.0
    f_x = c_x / np.tan(fov_x / 2.0)
    f_y = c_y / np.tan(fov_y / 2.0)

    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )

    return intrinsic_matrix


def project_world_to_image(
    trajectory_3D: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
) -> np.ndarray:
    """
    Takes an (N,3) list of 3D points
    intrinsic_matrix is (3,3)
    Returns an (N,3) list of 2D points on the camera plane

    :param trajectory_3D: (N,3) list of 3D points
    :type trajectory_3D: np.ndarray
    :param intrinsic_matrix: (3,3) intrinsics
    :type intrinsic_matrix: np.ndarray
    :param extrinsic_matrix: offsets to adjust the trajectory by
    :type extrinsic_matrix: np.ndarray
    :returns: (N,3) list of 2D points on the camera plane
    """
    # trajectory is (N, 3)
    # trajectory_3D_homo is (N, 4)
    # extrinsic_matrix is (4, 4)
    trajectory_3D_homo = np.array(
        [
            trajectory_3D[:, 0],
            trajectory_3D[:, 1],
            trajectory_3D[:, 2],
            np.ones_like(trajectory_3D[:, 0]),
        ]
    ).T
    # intrinsics_homo is (3, 4)
    intrinsics_homo = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # trajectory_2D_homo is (N, 3)
    trajectory_2D_homo = (
        intrinsics_homo @ extrinsic_matrix @ trajectory_3D_homo.T
    ).T
    # trajectory_2D is (N, 2)
    trajectory_2D = np.array(
        [
            trajectory_2D_homo[:, 0] / trajectory_2D_homo[:, 2],
            trajectory_2D_homo[:, 1] / trajectory_2D_homo[:, 2],
        ]
    ).T

    return trajectory_2D


def plot_steering_traj(
    frame_img: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    intrinsic_matrix: Optional[np.ndarray] = None,
    offsets: Tuple[float, float, float] = [0.0, 5.0, 3.0],
    method: str = "add_weighted",
    track: bool = True,
    line: bool = True,
    track_width: float = 4.5,
    samples: int = 10,
    thickness: int = 2,
    fov_x: float = 60,
    fov_y: float = 60,
) -> np.ndarray:
    """
    Coordinate Frames:
        3D:
            x: horizontal plane
            y: vertical into the ground
            z: depth into the camera
        2D Image:
            px: x axis
            py: y axis
        2D Trajectory:
            xt -> x
            yt -> z
            height -> y

      Plots a trajectory onto a given frame image.
    """
    assert method in ("overlay", "mask", "add_weighted")

    h, w = frame_img.shape[:2]

    if intrinsic_matrix is None:
        intrinsic_matrix = estimate_intrinsics(fov_x, fov_y, h, w)

    trajectory = interpolate_trajectory(trajectory, samples=samples)

    # trajectory_3D is (N,3)
    trajectory_3D = np.array(
        [
            -trajectory[:, 0],
            np.zeros_like(trajectory[:, 0]),
            trajectory[:, 1],
        ]
    ).T

    # trajectory_3D = np.concatenate(
    #     (
    #         [[0, 0, 0], ],
    #         trajectory_3D
    #     ),
    #     axis=0
    # )

    behind_cam = trajectory_3D[:, 2] - offsets[2] < 0.0

    trajectory_3D = trajectory_3D[~behind_cam]

    extrinsics = np.array(
        [
            [1, 0, 0, -offsets[0]],
            [0, 1, 0, -offsets[1]],
            [0, 0, 1, -offsets[2]],
            [0, 0, 0, 1],
        ]
    )

    trajectory_2D = project_world_to_image(
        trajectory_3D, intrinsic_matrix, extrinsics
    )

    # Filter out outliers
    trajectory_2D = trajectory_2D.astype(np.int16)

    rect_frame = np.zeros_like(frame_img)

    for point_index in range(1, trajectory_2D.shape[0]):
        px, py = trajectory_2D[point_index]
        px_p, py_p = trajectory_2D[point_index - 1]
        point_3D = trajectory_3D[point_index]
        prev_point_3D = trajectory_3D[point_index - 1]

        in_range = px_p in range(0, w) and py_p in range(0, h)

        if track and in_range:
            rect_coords_3D = get_rect_coords_3D(
                point_3D, prev_point_3D, track_width
            )
            behind_cam_poly = rect_coords_3D[:, 2] - offsets[2] < 0.0
            if behind_cam_poly.sum() == 0:
                rect_coords = project_world_to_image(
                    rect_coords_3D, intrinsic_matrix, extrinsics
                )
                rect_coords = rect_coords.astype(np.int32)
                rect_frame = cv2.fillPoly(
                    rect_frame, pts=[rect_coords], color=color
                )

        if line and in_range:
            frame_img = cv2.line(
                frame_img, (px_p, py_p), (px, py), color, thickness
            )
            rect_frame = cv2.line(
                rect_frame, (px_p, py_p), (px, py), color, thickness
            )

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_img[mask] = color
    elif method == "overlay":
        frame_img += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_img, 1.0, rect_frame, 0.5, 0.0, frame_img)
    return frame_img


def plot_bev_trajectory(
    trajectory: np.ndarray,
    frame_img: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    grid_range: float = 20,  # meters
):
    WIDTH, HEIGHT = frame_img.shape[1], frame_img.shape[0]
    traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    Z = trajectory[:, 1]
    X = trajectory[:, 0]

    X_min, X_max = -grid_range, grid_range
    # Z_min, Z_max = -RAN, RAN
    Z_min, Z_max = -0.1 * grid_range, grid_range
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    # X = (X - lb) / (ub - lb)
    # Z = (Z - lb) / (ub - lb)

    for traj_index in range(1, X.shape[0]):
        u = int(round(np.clip((X[traj_index] * (WIDTH - 1)), -1, WIDTH + 1)))
        v = int(round(np.clip((Z[traj_index] * (HEIGHT - 1)), -1, HEIGHT + 1)))
        u_p = int(
            round(np.clip((X[traj_index - 1] * (WIDTH - 1)), -1, WIDTH + 1))
        )
        v_p = int(
            round(np.clip((Z[traj_index - 1] * (HEIGHT - 1)), -1, HEIGHT + 1))
        )

        if u not in range(WIDTH) or v not in range(HEIGHT):
            continue

        traj_plot = cv2.circle(traj_plot, (u, v), thickness, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, thickness)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot


def get_rect_coords_3D(Pi: np.ndarray, Pj: np.ndarray, width: float):
    # Pi = Pi.reshape(4, 1)
    # Pj = Pj.reshape(4, 1)
    x_i, y_i = Pi[0], Pi[2]
    x_j, y_j = Pj[0], Pj[2]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0] = points_2D[index, 0]
        point_3D[2] = points_2D[index, 1]

        points_3D.append(point_3D)

    points_3D = np.array(points_3D)
    return points_3D


def get_rect_coords(x_i, y_i, x_j, y_j, width: float):
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return result


STEERING_IMG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "media/steering.png")
)


def plot_carstate_frame(
    frame_img,
    steering_gt=0.0,
    steering_pred=0.0,
    steering_img_path=STEERING_IMG_PATH,
):

    frame_img_steering = np.zeros_like(frame_img)

    steering_dim = round(frame_img.shape[0] * 0.2)

    # Draw Steering GT
    assert os.path.isfile(
        steering_img_path
    ), f"File not found: {steering_img_path}"
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_gt))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) - steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, 2.5, 0.0)

    # Draw Steering Pred
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_pred))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) + steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, -2.5, 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255, 0, 0)
    thickness = 1

    frame_img = cv2.putText(
        frame_img,
        "steering_gt: " + str(round(steering_gt, 2)),
        (10, 15),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    frame_img = cv2.putText(
        frame_img,
        "steer_pred: " + str(round(steering_pred, 2)),
        (10, 25),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame_img
