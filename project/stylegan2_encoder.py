import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class StyleGANEncoder(nn.Module):
    """Defines the encoder network for StyleGAN inversion."""

    def __init__(
        self,
        resolution=256,
        w_space_dim=512,
        image_channels=3,
        encoder_channels_base=64,
        encoder_channels_max=1024,
    ):
        super().__init__()
        # resolution = 256
        # w_space_dim = 512
        # image_channels = 3
        # encoder_channels_base = 64
        # encoder_channels_max = 1024

        self.init_res = 4
        self.resolution = resolution
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.encoder_channels_base = encoder_channels_base
        self.encoder_channels_max = encoder_channels_max

        # Blocks used in encoder.
        self.num_blocks = int(np.log2(resolution))  # 8

        # Layers used in generator.
        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2
        # (Pdb) self.num_layers -- 14

        in_channels = self.image_channels
        out_channels = self.encoder_channels_base
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                self.add_module(
                    f"block{block_idx}",
                    FirstBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                    ),
                )

            elif block_idx == self.num_blocks - 1:
                in_channels = in_channels * self.init_res * self.init_res
                out_channels = self.w_space_dim * 2 * block_idx
                self.add_module(
                    f"block{block_idx}",
                    LastBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                    ),
                )

            else:
                if in_channels != out_channels:
                    self.add_module(
                        f"block{block_idx}",
                        ResBlock1(
                            in_channels=in_channels,
                            out_channels=out_channels,
                        ),
                    )
                else:
                    self.add_module(
                        f"block{block_idx}",
                        ResBlock2(
                            in_channels=in_channels,
                            out_channels=out_channels,
                        ),
                    )
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.encoder_channels_max)

        self.downsample = AveragePoolingLayer()

    def forward(self, x):
        """
        The encoder takes images with `RGB` color channels and range [0, 1]
        as inputs, and encode the input images to W+ space of StyleGAN.
        """
        # (Pdb) x.size() -- [1, 3, 256, 256]

        # move x from [0.0, 1.0] to [-1.0, 1.0]
        x = 2.0 * (x - 0.5)
        # x = x.transpose(2, 0, 1)
        for block_idx in range(self.num_blocks):
            if 0 < block_idx < self.num_blocks - 1:
                x = self.downsample(x)
            x = self.__getattr__(f"block{block_idx}")(x)
        # (Pdb) x.size() -- [1, 7168]

        return x.view(1, 1, self.num_layers, 512)


class AveragePoolingLayer(nn.Module):
    """Implements the average pooling layer.

    Basically, this layer can be used to downsample feature maps from spatial
    domain.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        ksize = [self.scale_factor, self.scale_factor]
        strides = [self.scale_factor, self.scale_factor]
        return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


class BatchNormLayer(nn.Module):
    """Implements batch normalization layer."""

    def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon=1e-5):
        """Initializes with basic settings.

        Args:
          channels: Number of channels of the input tensor.
          gamma: Whether the scale (weight) of the affine mapping is learnable.
          beta: Whether the center (bias) of the affine mapping is learnable.
          decay: Decay factor for moving average operations in this layer.
          epsilon: A value added to the denominator for numerical stability.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=channels,
            affine=True,
            track_running_stats=True,
            momentum=1 - decay,
            eps=epsilon,
        )
        self.bn.weight.requires_grad = gamma
        self.bn.bias.requires_grad = beta

    def forward(self, x):
        return self.bn(x)


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.

    NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number, which is not trainable in
    this layer. However, the bias variable is trainable in this layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / np.sqrt(fan_in)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return x * self.scale + self.bias.view(1, -1, 1, 1)


class FirstBlock(nn.Module):
    """Implements the first block, which is a convolutional block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        wscale_gain=np.sqrt(2.0),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale = 1.0
        self.bn = BatchNormLayer(channels=out_channels)
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self = FirstBlock(
        #   (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #   (bn): BatchNormLayer(
        #     (bn): BatchNorm2d(64, eps=1e-05, momentum=0.09999999999999998, affine=True, track_running_stats=True)
        #   )
        #   (activate): LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # in_channels = 3
        # out_channels = 64
        # wscale_gain = 1.4142135623730951

    def forward(self, x):
        return self.activate(self.bn(self.conv(x) * self.scale))


class ResBlock1(nn.Module):
    """Implements the residual block1 -- with shortcut."""

    def __init__(
        self,
        in_channels,
        out_channels,
        wscale_gain=np.sqrt(2.0),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.scale = 1.0
        self.bn = BatchNormLayer(channels=out_channels)

        hidden_channels = min(in_channels, out_channels)

        # First convolutional block.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale1 = wscale_gain / np.sqrt(in_channels * 3 * 3)
        self.wscale1 = WScaleLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn1 = BatchNormLayer(channels=hidden_channels)

        # Second convolutional block.
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale2 = wscale_gain / np.sqrt(hidden_channels * 3 * 3)
        self.wscale2 = WScaleLayer(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn2 = BatchNormLayer(channels=out_channels)
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        y = self.activate(self.bn(self.conv(x) * self.scale))
        x = self.activate(self.bn1(self.wscale1(self.conv1(x) / self.scale1)))
        x = self.activate(self.bn2(self.wscale2(self.conv2(x) / self.scale2)))
        return x + y


class ResBlock2(nn.Module):
    """Implements the residual block1 -- without shortcut."""

    def __init__(
        self,
        in_channels,
        out_channels,
        wscale_gain=np.sqrt(2.0),
    ):
        super().__init__()

        hidden_channels = min(in_channels, out_channels)

        # First convolutional block.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale1 = wscale_gain / np.sqrt(in_channels * 3 * 3)
        self.wscale1 = WScaleLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn1 = BatchNormLayer(channels=hidden_channels)

        # Second convolutional block.
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scale2 = wscale_gain / np.sqrt(hidden_channels * 3 * 3)
        self.wscale2 = WScaleLayer(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            gain=wscale_gain,
        )
        self.bn2 = BatchNormLayer(channels=out_channels)
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        y = x
        x = self.activate(self.bn1(self.wscale1(self.conv1(x) / self.scale1)))
        x = self.activate(self.bn2(self.wscale2(self.conv2(x) / self.scale2)))
        return x + y


class LastBlock(nn.Module):
    """Implements the last block, which is a dense block."""

    def __init__(self, in_channels, out_channels, wscale_gain=1.0):
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=False
        )
        self.scale = wscale_gain / np.sqrt(in_channels)
        self.bn = BatchNormLayer(channels=out_channels)
        # self = LastBlock(
        #   (fc): Linear(in_features=16384, out_features=7168, bias=False)
        #   (bn): BatchNormLayer(
        #     (bn): BatchNorm2d(7168, eps=1e-05, momentum=0.09999999999999998, affine=True, track_running_stats=True)
        #   )
        # )
        # in_channels = 16384
        # out_channels = 7168
        # wscale_gain = 1.0

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x) * self.scale
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return self.bn(x).view(x.shape[0], x.shape[1])


if __name__ == "__main__":
    model = StyleGANEncoder()
    model.eval()

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        w = model(x)

    print(w.size())
