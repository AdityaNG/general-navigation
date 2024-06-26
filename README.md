# General Navigation Python Module

[![PyPI - Downloads](https://img.shields.io/pypi/dm/general_navigation)](https://pypi.org/project/general-navigation/)
[![PyPI - Version](https://img.shields.io/pypi/v/general-navigation)](https://pypi.org/project/general-navigation/)
[![codecov](https://codecov.io/gh/AdityaNG/general-navigation/branch/main/graph/badge.svg?token=general-navigation_token_here)](https://codecov.io/gh/AdityaNG/general-navigation)
[![CI](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml)
![GitHub License](https://img.shields.io/github/license/AdityaNG/general-navigation)

![Demo](https://raw.githubusercontent.com/AdityaNG/general-navigation/main/media/carla_demo.gif)

General Navigation Models based on GNM, ViNT, NoMaD as a pytorch repo installable from pip for quick and easy deployment. This repo provides functionality to load and run the models and use MPC to generate steering commands for your robot. If you find any functionality in this repo useful, consider citing it:

```
@misc{aditya2024general_navigation,
  title        = {General Navigation Python Module},
  author       = {Aditya, NG},
  howpublished = {\url{https://pypi.org/project/general-navigation/}},
  version      = {1.0.0},
  year         = {2024},
  note         = {Accessed: 2024-06-09}
}
```

## Install it from PyPI

[PyPi Link](https://pypi.org/project/general-navigation/)

Install our project from pip and quickly get started by trying it out on your own test video!

```bash
pip install general_navigation
python3 -m general_navigation --media media/test.mp4
```

If you want to connect with the Carla simulator, you will also need to seperately install carla
```bash
pip install carla==0.9.15  # Linux and Windows
pip install carla==0.9.5  # Mac
```

## Usage

[Documentation Link](https://adityang.github.io/general-navigation/)

Creating a pytorch instance of the model
```py
from general_navigation.models.factory import (
  get_default_config, get_model, get_weights
)

config = get_default_config()
model = get_model(config)
model = get_weights(config, model, device)
```

Using the command line tool for inference on a video
```bash
$ python3 -m general_navigation --help
usage: general_navigation [-h] [--device {auto,cuda,cpu}] [--media MEDIA]

options:
  -h, --help            show this help message and exit
  --device {auto,cuda,cpu}, -d {auto,cuda,cpu}
  --media MEDIA, -m MEDIA
                        File path, use camera index if you want to use the webcam
$ python3 -m general_navigation --media media/test.mp4
```

Using the command line tool for working with Carla simulator
```bash
$ python3 -m general_navigation.carla_run
```

## TODO

- [x] Import models from [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [x] Script to use specified video camera to feed video into the models
- [x] Visualize model's trajectory output
- [x] Arguments to CLI
- [x] Auto download model weights from [google drive](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg)
- [x] Demo video
- [x] PyPi release
- [x] Example usage
- [x] Carla Integration
- [x] MPC implementation
- [ ] Visualize MPC reconstructed trajectory
- [ ] Visualize all of the model's trajectories
- [x] Visualize Steering angle
- [ ] Fix scaling issue
- [ ] Intrinsic matrix as argument
- [x] Documentation: `mkdocs gh-deploy`
- [x] Linting fixes
- [ ] MyPy testing
- [ ] Add test cases for code coverage

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## References

- [general-navigation-models](https://general-navigation-models.github.io/)
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
