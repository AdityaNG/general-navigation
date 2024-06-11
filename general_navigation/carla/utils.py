import os
import signal
import subprocess
import time
from contextlib import contextmanager

from general_navigation.settings import settings


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
