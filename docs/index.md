# General Navigation

[![PyPI - Downloads](https://img.shields.io/pypi/dd/general-navigation)](https://pypi.org/project/general-navigation/)
[![PyPI - Version](https://img.shields.io/pypi/v/general-navigation)](https://pypi.org/project/general-navigation/)
[![codecov](https://codecov.io/gh/AdityaNG/general-navigation/branch/main/graph/badge.svg?token=general-navigation_token_here)](https://codecov.io/gh/AdityaNG/general-navigation)
[![CI](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml)
![GitHub License](https://img.shields.io/github/license/AdityaNG/general-navigation)

![Demo](https://raw.githubusercontent.com/AdityaNG/general-navigation/main/media/demo.gif)

General Navigation Models based on GNM, ViNT, NoMaD as a pytorch repo for quick and easy deployment.
Awesome general_navigation created by AdityaNG.

## Install it from PyPI

[PyPi Link](https://pypi.org/project/general-navigation/)

```bash
pip install general_navigation
```

## Usage

[Documentation Link](https://adityang.github.io/general-navigation/)

Creating a pytorch instance of the model
```py
from .models.factory import get_default_config, get_model, get_weights

config = get_default_config()
model = get_model(config)
model = get_weights(config, model, device)
```

Using the command line tool for inference
```bash
$ python3 -m general_navigation --help
CONFIG_DIR /home/aditya/miniconda3/envs/oi/lib/python3.11/site-packages/general_navigation/models/config
usage: general_navigation [-h] [--device {auto,cuda,cpu}] [--media MEDIA]

options:
  -h, --help            show this help message and exit
  --device {auto,cuda,cpu}, -d {auto,cuda,cpu}
  --media MEDIA, -m MEDIA
                        File path, use camera index if you want to use the webcam
$ python3 -m general_navigation --media media/test.mp4
```
