"""
GPT Vision to make Control Decisions in Carla
"""

import random
import time
import traceback

import carla
import cv2
import numpy as np
import torch

from general_navigation.carla.client import CarlaClient
from general_navigation.carla.utils import start_carla
from general_navigation.common import UIRecorder
from general_navigation.gpt.gpt_vision import GPTVision
from general_navigation.models.model_utils import generate_visual
from general_navigation.schema.environment import DroneControls


class RewardCalculator:
    def __init__(
        self,
        carla_client: CarlaClient,
    ):
        self.carla_client = carla_client
        self.client: carla.Client = carla_client.client
        self.carla_map = self.client.get_world().get_map()
        self.vehicle = self.client.get_vehicle()
        self.collision_event = None

        # Attach a collision sensor to the vehicle
        self.collision_sensor = self.carla_client.world.collision_sensor
        self.collision_sensor.listen(self.on_collision)

    def on_collision(self, event):
        self.collision_event = event

    def get_vehicle_state(self, drone_state):
        location = carla.Location(
            x=drone_state.position.x,
            y=drone_state.position.y,
            z=drone_state.position.z,
        )
        rotation = carla.Rotation(
            pitch=drone_state.rotation.pitch,
            yaw=drone_state.rotation.yaw,
            roll=drone_state.rotation.roll,
        )
        velocity = carla.Vector3D(
            x=drone_state.velocity.x,
            y=drone_state.velocity.y,
            z=drone_state.velocity.z,
        )
        acceleration = carla.Vector3D(
            x=drone_state.acceleration.x,
            y=drone_state.acceleration.y,
            z=drone_state.acceleration.z,
        )
        return {
            "location": location,
            "rotation": rotation,
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def calculate_speed(self, velocity):
        speed = (
            np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        )  # Convert m/s to km/h
        return speed

    def is_off_road(self, location):
        waypoint = self.carla_map.get_waypoint(location, project_to_road=False)
        return waypoint is None or waypoint.lane_type != carla.LaneType.Driving

    def check_lane_crossing(self, location):
        waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        if waypoint and waypoint.lane_type == carla.LaneType.Driving:
            left_wp = waypoint.get_left_lane()
            right_wp = waypoint.get_right_lane()
            left_x = left_wp.transform.location.x if left_wp else float("-inf")
            right_x = (
                right_wp.transform.location.x if right_wp else float("inf")
            )
            return location.x < left_x or location.x > right_x
        return True

    def check_traffic_signal_violation(self):
        traffic_light = self.vehicle.get_traffic_light()
        if traffic_light is not None:
            state = traffic_light.get_state()
            if (
                state == carla.TrafficLightState.Red
                and not self.vehicle.is_at_traffic_light()
            ):
                return True
        return False

    def compute_reward(self, drone_state):
        reward = 0.0

        # Speed Reward
        velocity = self.vehicle.get_velocity()
        speed = self.calculate_speed(velocity)
        target_speed = 50  # Target speed in km/h
        speed_penalty = -abs(target_speed - speed)
        reward += speed_penalty

        # Lane Keeping Reward
        location = carla.Location(
            x=drone_state.position.x,
            y=drone_state.position.y,
            z=drone_state.position.z,
        )
        if self.is_off_road(location):
            reward -= 100  # Large penalty for going off-road
        elif self.check_lane_crossing(location):
            reward -= 50  # Penalty for lane crossing

        # Traffic Signal Violation
        if self.check_traffic_signal_violation():
            reward -= 100  # Penalty for running red light

        # Collision Penalty
        if self.collision_event:
            reward -= 200  # Large penalty for collision
            self.collision_event = None  # Reset collision event

        # Optional: Distance Travelled Reward
        reward += speed * 0.1  # Small positive reward for moving forward

        return reward

    def cleanup(self):
        self.collision_sensor.stop()
        self.vehicle.destroy()


def main():
    client = CarlaClient()
    rec = UIRecorder()
    gpt: torch.nn.Module = GPTVision()
    drone_state = client.get_car_state()
    # last_update = drone_state.timestamp

    gpt_controls = DroneControls(
        trajectory=[(0, 0)],
        trajectory_mpc=[(0, 0)],
        speed=0.0,
        steer=0.0,
    )

    reward_calculator = RewardCalculator(client)
    optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-4)
    # gamma = 0.99  # Discount factor for future rewards

    for _ in range(gpt.context_size):
        drone_state = client.get_car_state(default=drone_state)
        gpt.append_context(drone_state)

    try:
        while True:
            start_time = time.time()
            client.game_loop()

            drone_state = client.get_car_state(default=drone_state)
            gpt.append_context(drone_state)
            state_tensor = gpt.context_to_torch()

            trajectory = gpt.forward(
                state_tensor
            )  # Forward pass to get controls
            randoom_selection = random.randint(0, trajectory.shape[0])
            trajectory_np = (
                trajectory[randoom_selection].cpu().detach().numpy()
            )
            accel, steer, traj_mpc = gpt.mpc.step(
                np.array(trajectory_np),
                drone_state.speed_ms(),
                drone_state.steering_angle,
            )

            gpt_controls.trajectory = trajectory_np
            gpt_controls.trajectory = gpt_controls.trajectory.tolist()
            gpt_controls.speed = gpt.speed
            gpt_controls.steer = steer
            gpt_controls.trajectory_mpc = traj_mpc

            client.set_car_controls(gpt_controls)

            # Calculate reward
            reward = reward_calculator.compute_reward(drone_state)
            print(f"Reward: {reward}")

            # TODO: Implement reward normalization if necessary

            # Compute loss and backpropagate
            action_log_probs = gpt_controls[
                "log_probs"
            ]  # Assume the model returns log probabilities of actions
            loss = -action_log_probs * reward
            loss = loss.sum()  # Sum up losses for the batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Visualization and recording
            visual = generate_visual(drone_state, gpt_controls)
            cv2.imshow("General Navigation", visual)
            rec.write(visual)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

            duration = time.time() - start_time
            fps = 1.0 / duration
            print(f"FPS: {fps} ; Time: {duration}")

    except KeyboardInterrupt:
        print("Land drone on keyboard interrupt, exiting...")
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        reward_calculator.cleanup()


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        print("Torch multiprocessing error, cannot proceed")
        exit()

    with start_carla():
        main()
