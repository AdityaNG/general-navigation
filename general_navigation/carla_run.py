"""
GPT Vision to make Control Decisions
"""

import os
import signal
import subprocess
import textwrap
import time
import traceback
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from general_navigation.carla.client import CarlaClient
from general_navigation.gpt.gpt_vision import GPTVision
from general_navigation.models.model_utils import (
    plot_bev_trajectory,
    plot_steering_traj,
)
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


@contextmanager
def start_carla():
    """
    Launch Carla using the command

    This function is to be used with the 'with' clause:
        ```py
        with start_carla():
            print("Carla is running")
        ```

        Once the function exits, carla is to shutdown
    """

    carla_script_path = os.path.join(
        settings.ui.CARLA_INSTALL_PATH, "CarlaUE4.sh"
    )
    assert os.path.isfile(
        carla_script_path
    ), f"File not found: {carla_script_path}"

    command = "CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -quality-level=Low -prefernvidia -ResX=10 -ResY=10"  # noqa
    try:
        # Start the Carla process
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=settings.ui.CARLA_INSTALL_PATH,
            preexec_fn=os.setsid,
        )
        time.sleep(5)
        yield process
    finally:
        # Ensure the Carla process is terminated upon exiting the context
        # Note: it appears that carla requires two of these to exit
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(1)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        process.wait()  # Wait for the process to properly terminate


class UIRecorder:

    def __init__(self) -> None:
        now = time.strftime("%Y-%m-%d--%H-%M-%S")
        self.path = os.path.join("test_media", f"DriveLLaVA-{now}.mp4")
        self.enabled = settings.ui.RECORDING_ENABLED
        self.fps = 30
        self.cap = None

    def write(self, image) -> None:
        if self.cap is None:
            size = image.shape[:2]
            self.cap = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size
            )

        self.cap.write(image)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


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
        speed=0.0,
        steer=0.0,
    )

    try:
        while True:
            start_time = time.time()
            client.game_loop()

            drone_state = client.get_car_state(default=drone_state)
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

            trajectory = np.array(gpt_controls.trajectory)
            plot_steering_traj(
                image_vis,
                trajectory,
                color=(255, 0, 0),
                track=True,
            )
            image_bev = plot_bev_trajectory(
                trajectory, image_vis, color=(255, 0, 0)
            )

            visual = np.hstack((image_vis, image_bev))
            print_text_image(visual, "GNM Controls")

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
