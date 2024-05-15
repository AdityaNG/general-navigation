import os
from typing import List

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

from general_navigation.visualizing.action_utils import plot_trajs_and_points
from general_navigation.visualizing.visualize_utils import from_numpy, to_numpy

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


def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


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


def interpolate_trajectory(trajectory, samples=1):
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


def plot_steering_traj(
    frame_center,
    trajectory,
    color=(255, 0, 0),
    intrinsic_matrix=None,
    DistCoef=None,
    offsets=[0.0, -1.2, -10.0],
    method="add_weighted",
    track=True,
    samples=10,
):
    assert method in ("overlay", "mask", "add_weighted")

    h, w = frame_center.shape[:2]

    trajectory = interpolate_trajectory(trajectory, samples=samples)

    if intrinsic_matrix is None:
        # intrinsic_matrix = np.array([
        #     [525.5030,         0,    333.4724],
        #     [0,         531.1660,    297.5747],
        #     [0,              0,    1.0],
        # ])
        intrinsic_matrix = np.array(
            [
                [525.5030, 0, w / 2],
                [0, 531.1660, h / 2],
                [0, 0, 1.0],
            ]
        )
    if DistCoef is None:
        DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, DistCoef, (w, h), 1, (w, h)
    )
    homo_cam_mat = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # rot = trajectory[0][:3,:3]
    # rot = np.eye(3,3)
    prev_point = None
    prev_point_3D = None
    rect_frame = np.zeros_like(frame_center)

    for trajectory_point in trajectory:
        p4d = np.ones((4, 1))
        p3d = np.array(
            [
                trajectory_point[0] * 1 - offsets[0],
                # trajectory_point[1] * 1 - offsets[1],
                -offsets[1],
                trajectory_point[1] * 1 - offsets[2],
            ]
        ).reshape((3, 1))
        # p3d = np.linalg.inv(rot) @ p3d
        p4d[:3, :] = p3d

        p2d = (homo_cam_mat) @ p4d
        if (
            p2d[2][0] != 0.0
            and not np.isnan(p2d).any()
            and not np.isinf(p2d).any()
        ):
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
            # frame_center = cv2.circle(frame_center, (px, py), 2, color, -1)
            if prev_point is not None:
                px_p, py_p = prev_point
                dist = ((px_p - px) ** 2 + (py_p - py) ** 2) ** 0.5
                if dist < 20:
                    if track:
                        rect_coords_3D = get_rect_coords_3D(p4d, prev_point_3D)
                        rect_coords = convert_3D_points_to_2D(
                            rect_coords_3D, homo_cam_mat
                        )
                        rect_frame = cv2.fillPoly(
                            rect_frame, pts=[rect_coords], color=color
                        )

                    frame_center = cv2.line(
                        frame_center, (px_p, py_p), (px, py), color, 2
                    )
                    # break

            prev_point = (px, py)
            prev_point_3D = p4d.copy()
        else:
            prev_point = None
            prev_point_3D = None

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_center[mask] = color
    elif method == "overlay":
        frame_center += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_center, 1.0, rect_frame, 0.2, 0.0, frame_center)
    return frame_center


def plot_bev_trajectory(trajectory, frame_center, color=(0, 255, 0)):
    WIDTH, HEIGHT = frame_center.shape[1], frame_center.shape[0]
    traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    Z = trajectory[:, 1]
    X = trajectory[:, 0]

    RAN = 20.0
    X_min, X_max = -RAN, RAN
    # Z_min, Z_max = -RAN, RAN
    Z_min, Z_max = -0.1 * RAN, RAN
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

        if u < 0 or u >= WIDTH or v < 0 or v >= HEIGHT:
            continue

        traj_plot = cv2.circle(traj_plot, (u, v), 2, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, 2)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot


def get_rect_coords_3D(Pi, Pj, width=1.5):
    x_i, y_i = Pi[0, 0], Pi[2, 0]
    x_j, y_j = Pj[0, 0], Pj[2, 0]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0, 0] = points_2D[index, 0]
        point_3D[2, 0] = points_2D[index, 1]

        points_3D.append(point_3D)

    return np.array(points_3D)


def get_rect_coords(x_i, y_i, x_j, y_j, width=2.83972):
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


def convert_3D_points_to_2D(points_3D, homo_cam_mat):
    points_2D = []
    for index in range(points_3D.shape[0]):
        p4d = points_3D[index]
        p2d = (homo_cam_mat) @ p4d
        px, py = 0, 0
        if p2d[2][0] != 0.0:
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])

        points_2D.append([px, py])

    return np.array(points_2D)


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
