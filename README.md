# General Navigation

[![codecov](https://codecov.io/gh/AdityaNG/general-navigation/branch/main/graph/badge.svg?token=general-navigation_token_here)](https://codecov.io/gh/AdityaNG/general-navigation)
[![CI](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/general-navigation/actions/workflows/main.yml)

General Navigation Models based on GNM, ViNT, NoMaD as a pytorch repo for quick and easy deployment.
Awesome general_navigation created by AdityaNG.

## Install it from PyPI

```bash
pip install general_navigation
```

## Usage

Creating a pytorch instance of the model
```py
from .models.factory import get_default_config, get_model, get_weights

config = get_default_config()
model = get_model(config)
model = get_weights(config, model, device)
```

Using the command line tool for inference
```bash
$ python -m general_navigation
```

## TODO

- [x] Import models from [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [x] Script to use specified video camera to feed video into the models
- [ ] Visualize model's trajectory output
- [ ] Arguments to CLI
- [ ] Auto download model weights from [google drive](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg)
- [ ] PyPi release
- [ ] Example usage
- [ ] Carla Integration
- [ ] Fix scaling issue
- [ ] Intrinsic matrix as argument
- [ ] Linting fixes
- [ ] Add test cases for code coverage

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## References

- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy)