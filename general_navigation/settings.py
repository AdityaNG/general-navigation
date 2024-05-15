import os


def str_to_int(value: str, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def str_to_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    return False


class DroneSettings:
    DRONE_NAME: str = os.getenv("DRONE_NAME", "airsim")


class UISettings:
    UI_ENABLED: bool = str_to_bool(os.getenv("UI_ENABLED", "True"))
    RECORDING_ENABLED: bool = str_to_bool(
        os.getenv("RECORDING_ENABLED", "False")
    )
    CARLA_INSTALL_PATH: str = os.getenv(
        "CARLA_INSTALL_PATH", os.path.expanduser("~/Apps/CARLA")
    )


class Settings:
    drone: DroneSettings = DroneSettings()  # type: ignore[call-arg]
    ui: UISettings = UISettings()  # type: ignore[call-arg]


settings = Settings()
