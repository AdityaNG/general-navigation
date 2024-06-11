"""
GPT Vision to make Control Decisions in Carla
"""

import time
import traceback

import cv2
import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from general_navigation.carla.client import CarlaClient
from general_navigation.carla.utils import start_carla
from general_navigation.common import UIRecorder
from general_navigation.gpt.gpt_vision import GPTVision
from general_navigation.models.model_utils import generate_visual
from general_navigation.schema.environment import DroneControls
from general_navigation.settings import settings


def async_gpt(gpt_input_q: Queue, gpt_output_q: Queue, gpt: GPTVision):

    while True:
        try:
            data = gpt_input_q.get()
            drone_state = data
            gpt_controls = gpt.step(drone_state)
            gpt_controls_dict = gpt_controls.model_dump()
            gpt_output_q.put(gpt_controls_dict)
        except Exception as ex:
            print("Exception while calling GPT", ex)
            traceback.print_exc()


def select_trajectory_index(trajectory_templates, trajectory_index):
    template_trajectory = trajectory_templates[trajectory_index]
    template_trajectory_3d = np.zeros((settings.system.TRAJECTORY_SIZE, 3))
    template_trajectory_3d[:, 0] = template_trajectory[:, 0]
    template_trajectory_3d[:, 2] = template_trajectory[:, 1]

    return template_trajectory_3d


def main():  # pragma: no cover
    """
    Use the Image from the sim and the map to feed as input to GPT
    Return the vehicle controls to the sim and step
    """

    device = "auto"
    client = CarlaClient()
    rec = UIRecorder()
    gpt = GPTVision()

    drone_state = client.get_car_state()
    # image = drone_state.image.cv_image()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    last_update = drone_state.timestamp

    gpt_input_q = Queue(maxsize=1)
    gpt_output_q = Queue(maxsize=1)

    gpt_process = Process(
        target=async_gpt, args=(gpt_input_q, gpt_output_q, gpt)
    )
    gpt_process.daemon = True
    gpt_process.start()

    gpt_controls = DroneControls(
        trajectory=[
            (0, 0),
        ],
        trajectory_mpc=[
            (0, 0),
        ],
        speed=0.0,
        steer=0.0,
    )

    try:
        while True:
            start_time = time.time()
            client.game_loop()

            drone_state = client.get_car_state(default=drone_state)

            if gpt_input_q.empty():
                data = drone_state
                gpt_input_q.put(data)

            # Get GPT Controls
            if not gpt_output_q.empty() or (
                settings.system.GPT_WAIT
                and time.time() * 1000 - last_update > 10.0 * 1000
            ):
                gpt_controls_dict = gpt_output_q.get()
                gpt_controls = DroneControls(**gpt_controls_dict)
                client.set_car_controls(gpt_controls)
                last_update = gpt_controls.timestamp

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
        gpt_process.kill()


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        print("Torch multiprocessing error, cannot proceed")
        exit()

    with start_carla():
        main()
