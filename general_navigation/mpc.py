import time

import numpy as np
from scipy.optimize import minimize
from simple_pid import PID

from general_navigation.common import MAX_STEER, STEERING_RATIO, WHEEL_BASE


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
        self.steering_pred_list = np.zeros(
            self.horizon,
        )
        self.last_accel = 0.0
        self.last_update_time = time.time()

    def step(self, trajectory, set_speed):
        now = time.time()
        accel, steer = 0.0, 0.0  # Default to no action

        ###########################################
        # Time stamps in seconds
        now = time.time()
        ###########################################

        self.trajectory = trajectory

        #####################################
        # Lateral

        self.steering_pred_list = MPC_run(
            self.trajectory.copy(),
            set_speed,
            self.time_step,
            WHEEL_BASE,
            STEERING_RATIO,
            self.horizon,
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
        return accel, steer


def MPC_run(trajectory, velocity, time_step, WHEEL_BASE, STEERING_RATIO, N):
    """
    TODO: Optimize this function
    Currently, it takes about 60ms to 80ms to run
    Optimization ideas:
            - Discretize the trajectory and hash the results values
            - Use a faster solver
            - Precompute the trajectory cache using slower solver offline,
                use faster solver offline for cache miss
            - Unsure we're solving using all cores
    """
    K = 0.00005  # tuning parameter for the cost function
    # K is the inverse agressiveness of the steering input
    # Smaller K values correspond to more aggressive steering
    # Larger K values correspond to less aggressive steering

    result_trajectory = np.zeros(N)

    try:
        dt = time_step
        trajectory_interp = traverse_trajectory(
            trajectory.copy(), velocity * dt
        )

        print("trajectory", trajectory.shape)
        print("trajectory_interp", trajectory_interp.shape)

        assert trajectory_interp.shape[0] > 1, "Not enough points"

        # define the model
        def bicycle_model(x, u):
            delta = np.radians(u) / STEERING_RATIO
            x_next = np.zeros(4)
            x_next[2] = x[2] + (
                velocity / WHEEL_BASE * np.tan(delta) * dt
            )  # theta
            x_next[0] = x[0] + (velocity * np.cos(x_next[2]) * dt)  # x pos
            x_next[1] = x[1] + (velocity * np.sin(x_next[2]) * dt)  # y pos
            x_next[3] = velocity
            return x_next

        # define the cost function
        def cost(u, x, x_des):
            cost_val = 0.0
            for i in range(N):
                x = bicycle_model(x, u[i])

                # Compute cost for each point on the trajectory
                cost_val += (
                    (x[0] - x_des[i, 0]) ** 2
                    + (x[1] - x_des[i, 1]) ** 2
                    + K * u[i] ** 2
                )

            return cost_val

        # initial state and input sequence
        # TODO: incorporate current steering angle
        x0 = np.array(
            [trajectory_interp[0, 0], trajectory_interp[0, 1], 0.0, velocity]
        )
        # TODO: incorporate current trajectory
        u0 = np.zeros(N)

        # bounds on the steering angle
        bounds = []
        for _ in range(N):
            bounds.append((-MAX_STEER, MAX_STEER))

        # optimize the cost function
        res = minimize(
            cost,
            u0,
            args=(x0, trajectory_interp.copy()),
            method="SLSQP",
            bounds=bounds,
            options=dict(maxiter=100),
        )

        u_opt = res.x
        result_trajectory = u_opt
    except Exception as ex:
        print("Error: ", ex)
    finally:
        return result_trajectory


def traverse_trajectory(traj, D):
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
        if dist + traj_dist > D:
            traj_interp.append(traj[traj_i - 1])
            dist = traj_dist
        else:
            dist += traj_dist
        total_dist += traj_dist

    return np.array(traj_interp)
