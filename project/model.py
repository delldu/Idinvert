"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 10日 星期二 15:49:52 CST
# ***
# ************************************************************************************/
#

import os
import pdb  # For debug

import torch

from stylegan2_encoder import StyleGANEncoder
from stylegan2_decoder import StyleGANDecoder
from stylegan2_refiner import VGG16, StyleGANRefiner


def model_load(model, path, prefix=""):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if len(prefix) > 0 and not n.startswith(prefix):
            continue
        n = n.replace(prefix, "")
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


def get_encoder(checkpoint):
    """Create model."""

    model = StyleGANEncoder()
    model_load(model, checkpoint, prefix="")

    return model


def get_decoder(checkpoint):
    """Create model."""

    model = StyleGANDecoder()
    model_load(model, checkpoint, prefix="synthesis.")

    return model


def get_vgg16(checkpoint):
    """Create model."""

    model = VGG16()
    model_load(model, checkpoint, prefix="")

    return model


def get_refiner():
    model = StyleGANRefiner()
    # StyleGANRefiner auto loading checkpoint
    return model


def model_device():
    """Please call after model_setenv."""

    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random

    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    if os.environ["DEVICE"] == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
