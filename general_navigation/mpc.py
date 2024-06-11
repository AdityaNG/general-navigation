import time
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from simple_pid import PID

from general_navigation.common import MAX_STEER, STEERING_RATIO, WHEEL_BASE
from general_navigation.models.model_utils import interpolate_trajectory


class MPC:
    def __init__(
        self,
        desired_velocity,
        time_step,
        horizon,
    ) -> None:
        self.desired_velocity = desired_velocity
        self.time_step = time_step
        self.horizon = horizon

        self.speed_pid = PID(0.1, 0.0, 0.0, setpoint=self.desired_velocity)
        self.speed_pid.output_limits = (-0.5, 0.05)
        self.speed_pid.sample_time = self.time_step  # seconds

        self.traj_pred = np.zeros((8, 2))
        self.last_accel = 0.0
        self.last_update_time = time.time()
        self.steering_pred_list = np.zeros(self.horizon)
        self.trajectory_list = np.zeros((self.horizon, 2))
        # self.inverse_agressiveness = 0.000075
        self.inverse_agressiveness = 0.0000075
        self.interpolate_samples = 8

    def step(self, trajectory, set_speed, current_steering):
        now = time.time()
        accel, steer = 0.0, 0.0  # Default to no action

        ###########################################
        # Time stamps in seconds
        now = time.time()
        ###########################################

        self.trajectory = trajectory

        #####################################
        # Lateral

        self.trajectory_list, self.steering_pred_list = mpc_run(
            self.trajectory.copy(),
            set_speed,
            self.time_step,
            WHEEL_BASE,
            STEERING_RATIO,
            self.horizon,
            current_steering,
            MAX_STEER,
            self.inverse_agressiveness,
            self.interpolate_samples,
            self.steering_pred_list,
            self.trajectory_list,
        )

        # Autopilot steering command (-1.0, +1.0)
        steer = self.steering_pred_list[0] / float(MAX_STEER)

        #####################################
        # Longitudinal

        accel = self.speed_pid(set_speed)
        accel = np.clip(accel, -1.0, 0.1)
        #####################################
        self.last_update_time = now

        # Update the last accel
        self.last_accel = accel
        return accel, steer, self.trajectory_list


def mpc_run(
    trajectory: np.ndarray,
    velocity: float,
    time_step: float,
    wheel_base: float,
    steering_ratio: float,
    horizon: int,
    current_steering: float,
    max_steer: float,
    inverse_agressiveness: float,
    interpolate_samples: int,
    steering_history: np.ndarray,
    result_trajectory: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MPC to produce the next desired streering angle.
    It also takes the prevous steering sequence and the vehicle's current
    steering angle into account to ensure that the next resulting steering
    angle sequence is as close as possible to what is currently planned
    while ensuring that trajectory is pursued

    :param np.ndarray trajectory: desired trajectory to be taken
    :param float velocity: velocity in meters per second
    :param float time_step: time delta between frames in seconds
    :param float wheel_base: vehicle wheel base in meters
    :param float steering_ratio: steering ratio of the vehicle
    :param int horizon: number of frames into the future to optimize for
    :param float current_steering: the vehicle's current steering angle
    :param float max_steer: maximum steering angle in degrees
    :param float inverse_agressiveness: tuning parameter for the controller
    :param int interpolate_samples: tuning parameter interpolation
    :param np.ndarray steering_history: the previously planned steering \
        sequence
    :param np.ndarray result_trajectory: the previously planned trajectory \
        resulting from the previous steering sequence

    :return np.ndarray steering_history: the newly planned steering sequence \
        defaults to the input steering_history if error
    :return np.ndarray result_trajectory: the newly planned trajectory \
        defaults to the input result_trajectory if error
    """
    try:
        #######################################################################
        # Setup params

        K = inverse_agressiveness  # tuning parameter for the cost function
        # K is the inverse agressiveness of the steering input
        # Smaller K values correspond to more aggressive steering
        # Larger K values correspond to less aggressive steering

        dt = time_step

        # Ensure you have sufficient points
        trajectory = interpolate_trajectory(
            trajectory, samples=interpolate_samples
        )
        # Ensure all points are seperated by the distance step
        trajectory_interp = traverse_trajectory(
            trajectory.copy(), velocity * dt
        )
        trajectory_interp = trajectory_interp[:, [1, 0]]
        trajectory_interp = trajectory_interp[0:horizon]

        assert trajectory_interp.shape[0] >= horizon, (
            f"Not enough points, expected at least {horizon}, "
            f"got {trajectory_interp.shape[0]}"
        )
        #######################################################################
        # define the model

        def bicycle_model(x, u):
            """
            x <- (X, Y, theta, velocity)
            u <- steering angle
            """
            delta = np.radians(u) / steering_ratio
            x_next = np.zeros(4)
            x_next[2] = x[2] + (
                velocity / wheel_base * np.tan(delta) * dt
            )  # theta
            x_next[0] = x[0] + (velocity * np.cos(x_next[2]) * dt)  # x pos
            x_next[1] = x[1] + (velocity * np.sin(x_next[2]) * dt)  # y pos
            x_next[3] = velocity
            return x_next

        #######################################################################
        # define the cost function

        def cost(u, x, x_des):
            """
            u[i] <- steering angle for step i
            x <- (X, Y, theta, velocity) initial state
            x_des[i] <- (X, Y, theta, velocity) desired state at step i
            """
            cost_val = 0.0
            for i in range(horizon):
                x = bicycle_model(x, u[i])

                # Compute cost for each point on the trajectory
                cost_val += (
                    (x[0] - x_des[i, 0]) ** 2
                    + (x[1] - x_des[i, 1]) ** 2
                    + K * u[i] ** 2
                )

            return cost_val

        #######################################################################
        # Optimizer Setup

        # initial state and input sequence
        x0 = np.array(
            [
                trajectory_interp[0, 0],
                trajectory_interp[0, 1],
                current_steering,
                velocity,
            ]
        )
        # Incorporate current trajectory
        u0 = steering_history.copy()
        # bounds on the steering angle
        bounds = []
        for _ in range(horizon):
            bounds.append((-max_steer, max_steer))
        #######################################################################
        # Optimize the cost function
        res = minimize(
            cost,
            u0,
            args=(x0, trajectory_interp.copy()),
            method="SLSQP",
            bounds=bounds,
            options=dict(maxiter=20),
        )

        u_opt = res.x
        steering_history = u_opt
        #######################################################################
        # Compute the new trajectory that results from the steering sequence
        result_trajectory = np.zeros((horizon, 4))
        result_trajectory[0, :] = x0
        result_trajectory[:, 3] = velocity
        for i in range(1, horizon):
            result_trajectory[i] = bicycle_model(
                result_trajectory[i - 1], steering_history[i]
            )
        result_trajectory = np.array(result_trajectory)
        result_trajectory = result_trajectory[:, :2]
        result_trajectory = result_trajectory[:, [1, 0]]
        #######################################################################
    except Exception as ex:
        print("Error: ", ex)
    finally:
        return result_trajectory, steering_history


def traverse_trajectory(
    traj: np.ndarray,
    distance: float,
) -> np.ndarray:
    """
    Takes a (N, 2) trajectory as input and produces an (M, 2) trajectory such
    that the new trajectory's adjacent points are all seperated by the
    specified distance.

    :param np.ndarray traj: input trajectory in meters (N, 2)
    :param float distance: split distance

    :return np.ndarray split_traj: trajectory where all adjacent points are \
        seperated by the specified distance
    """
    traj_interp = [
        traj[0],
    ]
    dist = 0.0
    total_dist = 0.0
    for traj_i in range(1, traj.shape[0]):
        traj_dist = (
            (traj[traj_i, 0] - traj[traj_i - 1, 0]) ** 2
            + (traj[traj_i, 1] - traj[traj_i - 1, 1]) ** 2
        ) ** 0.5
        if dist + traj_dist > distance:
            traj_interp.append(traj[traj_i - 1])
            dist = traj_dist
        else:
            dist += traj_dist
        total_dist += traj_dist

    return np.array(traj_interp)
