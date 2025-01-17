"""Model predict."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 10日 星期二 15:49:52 CST
# ***
# ************************************************************************************/
#
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pdb

from tqdm import tqdm

from stylegan2_encoder import StyleGANEncoder
from stylegan2_decoder import StyleGANDecoder


MEAN_STATS = (103.939, 116.779, 123.68)


class VGG16(nn.Sequential):
    """Defines the VGG16 structure as the perceptual network.

    This models takes `RGB` images with pixel range [-1, 1] and data format `NCHW`
    as raw inputs. This following operations will be performed to preprocess the
    inputs (as defined in `keras.applications.imagenet_utils.preprocess_input`):
    (1) Shift pixel range to [0, 255].
    (3) Change channel order to `BGR`.
    (4) Subtract the statistical mean.

    NOTE: The three fully connected layers on top of the model are dropped.
    """

    def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
        """Defines the network structure.

        Args:
          output_layer_idx: Index of layer whose output will be used as perceptual
            feature. (default: 23, which is the `block4_conv3` layer activated by
            `ReLU` function)
        """
        sequence = OrderedDict(
            {
                "layer0": nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                "layer1": nn.ReLU(inplace=True),
                "layer2": nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                "layer3": nn.ReLU(inplace=True),
                "layer4": nn.MaxPool2d(kernel_size=2, stride=2),
                "layer5": nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                "layer6": nn.ReLU(inplace=True),
                "layer7": nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                "layer8": nn.ReLU(inplace=True),
                "layer9": nn.MaxPool2d(kernel_size=2, stride=2),
                "layer10": nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                "layer11": nn.ReLU(inplace=True),
                "layer12": nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                "layer13": nn.ReLU(inplace=True),
                "layer14": nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                "layer15": nn.ReLU(inplace=True),
                "layer16": nn.MaxPool2d(kernel_size=2, stride=2),
                "layer17": nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                "layer18": nn.ReLU(inplace=True),
                "layer19": nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                "layer20": nn.ReLU(inplace=True),
                "layer21": nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                "layer22": nn.ReLU(inplace=True),
                "layer23": nn.MaxPool2d(kernel_size=2, stride=2),
                "layer24": nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                "layer25": nn.ReLU(inplace=True),
                "layer26": nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                "layer27": nn.ReLU(inplace=True),
                "layer28": nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                "layer29": nn.ReLU(inplace=True),
                "layer30": nn.MaxPool2d(kernel_size=2, stride=2),
            }
        )
        self.output_layer_idx = output_layer_idx
        self.min_val = min_val
        self.max_val = max_val
        self.mean = torch.from_numpy(np.array(MEAN_STATS)).view(1, 3, 1, 1)
        self.mean = self.mean.type(torch.FloatTensor)

        super().__init__(sequence)

    def forward(self, x):
        # x.shape -- [1, 3, 256, 256], x.min(), x.max() -- -0.9922, 0.9922
        x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
        x = x[:, [2, 1, 0], :, :]
        x = x - self.mean.to(x.device)
        for i in range(self.output_layer_idx):
            x = self.__getattr__(f"layer{i}")(x)
        return x


def get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


class StyleGANRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.learning_rate = 1e-2
        self.decoder_loss_weight = 1.0
        self.vgg16_loss_weight = 5e-5
        self.encoder_loss_weight = 2.0

        self.encoder = StyleGANEncoder()
        self.decoder = StyleGANDecoder()
        self.vgg16 = VGG16()

    def forward(self, x):
        x.requires_grad = False

        # x.size() -- [1, 3, 256, 256]
        with torch.no_grad():
            wcode = self.encoder(x)

        wcode.requires_grad = True

        optimizer = optim.Adam([wcode], lr=self.learning_rate)

        progress = tqdm(range(1, self.epochs + 1), leave=True)
        for step in progress:
            loss = 0.0

            loss_msg = f"Loss: "

            # Pixel/Decode loss
            y = self.decoder(wcode)
            loss_pixel = torch.mean((x - y) ** 2)
            loss = loss + loss_pixel * self.decoder_loss_weight
            loss_msg += f"decoder {get_tensor_value(loss_pixel):6.4f}, "

            # VGG16 loss
            x_feat = self.vgg16(x)
            y_feat = self.vgg16(y)
            loss_feat = torch.mean((x_feat - y_feat) ** 2)
            loss = loss + loss_feat * self.vgg16_loss_weight
            loss_msg += f"perception {get_tensor_value(loss_feat):8.3f}, "

            # Encode Loss
            wcode_rec = self.encoder(y)
            loss_reg = torch.mean((wcode - wcode_rec) ** 2)
            loss = loss + loss_reg * self.encoder_loss_weight
            loss_msg += f"encoder {get_tensor_value(loss_reg):6.4f}, "

            loss_msg += f"total {get_tensor_value(loss):6.4f}"
            progress.set_description_str(loss_msg)

            # Do optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return wcode


class Attribute(object):
    '''Get Attribute -- age, express, gender, glass, pose'''

    def __init__(self):
        self.attributes = torch.load("models/stylegan2_feature.pth")

    def valid_name(self, name):
        assert name +".layer" in self.attributes.keys()

    def get_layer(self, name):
        self.valid_name(name)
        return self.attributes[name + ".layer"]

    def get_attrs(self, name):
        self.valid_name(name)
        return self.attributes[name + ".attrs"]

    def get_wcodes(self, wcode, name):
        wcode = wcode.view(1, 1, 14, 512)
        
        distance = torch.Tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).to(wcode.device)
        distance = distance.view(1, 7, 1, 1)

        attrs = self.get_attrs(name).to(wcode.device)
        attrs = attrs.view(1, 1, 14, 512)
        
        layers = self.get_layer(name).to(wcode.device)


        x = wcode + distance * attrs
        y = wcode.repeat(1, 7, 1, 1)

        select = torch.zeros(y.shape, dtype=bool)
        select[:, :, layers] = True

        return torch.where(select.to(wcode.device), x, y)
