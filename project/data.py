"""Data loader."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 10日 星期二 15:49:52 CST
# ***
# ************************************************************************************/
#

import pdb  # For debug

import torch
import torchvision.utils as utils
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Define Train/Test Dataset Root
# ***
# ************************************************************************************/
#

def grid_image(tensor_list, nrow=3):
    grid = utils.make_grid(torch.cat(tensor_list, dim=0), nrow=nrow)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    image = Image.fromarray(ndarr)
    return image
