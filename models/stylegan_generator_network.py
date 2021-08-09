# python 3.7
"""Contains the implementation of generator described in StyleGAN.

For more details, please check the original paper:
https://arxiv.org/pdf/1812.04948.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

__all__ = ["StyleGANGeneratorNet"]

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Fused-scale options allowed.
_FUSED_SCALE_OPTIONS_ALLOWED = [True, False, "auto"]


class StyleGANGeneratorNet(nn.Module):
    """Defines the generator network in StyleGAN.

    NOTE: the generated images are with `RGB` color channels and range [-1, 1].
    """

    def __init__(
        self,
        resolution,
        z_space_dim=512,
        w_space_dim=512,
        num_mapping_layers=8,
        image_channels=3,
        truncation_psi=0.7,
        truncation_layers=8,
    ):
        super().__init__()
        # resolution = 256
        # z_space_dim = 512
        # w_space_dim = 512
        # num_mapping_layers = 8
        # image_channels = 3
        # truncation_psi = 0.7
        # truncation_layers = 8

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(
                f"Invalid resolution: {resolution}!\n"
                f"Resolutions allowed: {_RESOLUTIONS_ALLOWED}."
            )


        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.num_mapping_layers = num_mapping_layers
        self.image_channels = image_channels
        self.truncation_psi = truncation_psi
        self.truncation_layers = truncation_layers

        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModule(
            input_space_dim=self.z_space_dim,
            hidden_space_dim=512,
            final_space_dim=mapping_space_dim,
            num_layers=self.num_mapping_layers,
        )
        self.truncation = TruncationModule(
            num_layers=self.num_layers,
            w_space_dim=self.w_space_dim,
            truncation_psi=self.truncation_psi,
            truncation_layers=self.truncation_layers,
        )
        self.synthesis = SynthesisModule(
            init_resolution=self.init_res,
            resolution=self.resolution,
            w_space_dim=self.w_space_dim,
            image_channels=self.image_channels,
        )


class MappingModule(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(
        self,
        input_space_dim=512,
        hidden_space_dim=512,
        final_space_dim=512,
        num_layers=8,
    ):
        super().__init__()

        self.input_space_dim = input_space_dim
        self.num_layers = num_layers

        self.norm = PixelNormLayer()

        for i in range(num_layers):
            dim_mul = 1
            in_dim = input_space_dim * dim_mul if i == 0 else hidden_space_dim
            out_dim = final_space_dim if i == (num_layers - 1) else hidden_space_dim
            self.add_module(f"dense{i}", DenseBlock(in_dim, out_dim))
        # input_space_dim = 512
        # hidden_space_dim = 512
        # final_space_dim = 7168
        # num_layers = 8

    def forward(self, z, l=None):
        w = self.norm(z)
        # num_layers = 8
        for i in range(self.num_layers):
            w = self.__getattr__(f"dense{i}")(w)
        return w


class TruncationModule(nn.Module):
    """Implements the truncation module."""

    def __init__(
        self,
        num_layers,
        w_space_dim=512,
        truncation_psi=0.7,
        truncation_layers=8,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        # num_layers = 14
        # w_space_dim = 512
        # truncation_psi = 0.7
        # truncation_layers = 8

        if truncation_psi is not None and truncation_layers is not None:
            self.use_truncation = True
        else:
            self.use_truncation = False
            truncation_psi = 1.0
            truncation_layers = 0

        self.register_buffer("w_avg", torch.zeros(w_space_dim))

        layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
        coefs = np.ones_like(layer_idx, dtype=np.float32)
        coefs[layer_idx < truncation_layers] *= truncation_psi
        self.register_buffer("truncation", torch.from_numpy(coefs))

    def forward(self, w):
        pdb.set_trace()
        # xxxx8888
        print(
            "TruncationModule: w.ndim == ",
            w.ndim,
            "self.use_truncation == ",
            self.use_truncation,
        )
        if w.ndim == 2:
            pdb.set_trace()
            assert w.shape[1] == self.w_space_dim * self.num_layers
            w = w.view(-1, self.num_layers, self.w_space_dim)
        assert w.ndim == 3 and w.shape[1:] == (self.num_layers, self.w_space_dim)
        if self.use_truncation:
            pdb.set_trace()

            w_avg = self.w_avg.view(1, 1, self.w_space_dim)
            w = w_avg + (w - w_avg) * self.truncation
        return w


class SynthesisModule(nn.Module):
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(
        self,
        init_resolution=4,
        resolution=1024,
        w_space_dim=512,
        image_channels=3,
    ):
        super().__init__()
        # init_resolution = 4
        # resolution = 256
        # w_space_dim = 512
        # image_channels = 3

        self.init_res = init_resolution
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.w_space_dim = w_space_dim

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # Level of detail (used for progressive training).
        self.lod = nn.Parameter(torch.zeros(()))

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            if res == self.init_res:
                self.add_module(
                    f"layer{2 * block_idx}",
                    FirstConvBlock(
                        init_resolution=self.init_res,
                        channels=self.get_nf(res),
                        w_space_dim=self.w_space_dim,
                    ),
                )
            else:
                if (res >= 128):
                    self.add_module(
                        f"layer{2 * block_idx}",
                        UpConvBlockScale(
                            resolution=res,
                            in_channels=self.get_nf(res // 2),
                            out_channels=self.get_nf(res),
                            w_space_dim=self.w_space_dim,
                        ),
                    )
                else:
                    self.add_module(
                        f"layer{2 * block_idx}",
                        UpConvBlock(
                            resolution=res,
                            in_channels=self.get_nf(res // 2),
                            out_channels=self.get_nf(res),
                            w_space_dim=self.w_space_dim,
                        ),
                    )

            # Second convolution layer for each resolution.
            self.add_module(
                f"layer{2 * block_idx + 1}",
                ConvBlock(
                    resolution=res,
                    in_channels=self.get_nf(res),
                    out_channels=self.get_nf(res),
                    w_space_dim=self.w_space_dim,
                ),
            )

            # Output convolution layer for each resolution.
            self.add_module(
                f"output{block_idx}",
                LastConvBlock(
                    in_channels=self.get_nf(res), out_channels=image_channels
                ),
            )

        self.upsample = ResolutionScalingLayer()
        self.final_activate = nn.Tanh()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min((16<<10) // res, 512)

    def forward(self, w):
        lod = self.lod.cpu().tolist()
        # (Pdb) self.lod -- Parameter containing: tensor(0., device='cuda:0', requires_grad=True)
        # self.init_res_log2 -- 2
        # self.final_res_log2 -- 8
        # xxxx8888
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            if res_log2 + lod <= self.final_res_log2:
                block_idx = res_log2 - self.init_res_log2
                if block_idx == 0:
                    x = self.__getattr__(f"layer{2 * block_idx}")(w[:, 2 * block_idx])
                else:
                    x = self.__getattr__(f"layer{2 * block_idx}")(
                        x, w[:, 2 * block_idx]
                    )
                x = self.__getattr__(f"layer{2 * block_idx + 1}")(
                    x, w[:, 2 * block_idx + 1]
                )
                image = self.__getattr__(f"output{block_idx}")(x)
            else:
                image = self.upsample(image)
        image = self.final_activate(image)
        return image


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        x = x / torch.sqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.eps)
        return x


class ResolutionScalingLayer(nn.Module):
    """Implements the resolution scaling layer.

    Basically, this layer can be used to upsample feature maps from spatial domain
    with nearest neighbor interpolation.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")


class BlurLayer(nn.Module):
    """Implements the blur layer."""

    def __init__(self, channels, kernel=(1, 2, 1), normalize=True, flip=False):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer("kernel", torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class NoiseApplyingLayer(nn.Module):
    """Implements the noise applying layer."""

    def __init__(self, resolution, channels):
        super().__init__()
        self.res = resolution
        self.register_buffer("noise", torch.randn(1, 1, self.res, self.res))
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return x + self.noise * self.weight.view(1, -1, 1, 1)


class StyleModulationLayer(nn.Module):
    """Implements the style modulation layer."""

    def __init__(self, channels, w_space_dim=512):
        super().__init__()
        self.channels = channels
        self.w_space_dim = w_space_dim
        self.dense = DenseBlock(
            in_channels=w_space_dim,
            out_channels=channels * 2,
            wscale_gain=1.0,
            wscale_lr_multiplier=1.0,
            activation_type="linear",
        )
        # channels = 512
        # w_space_dim = 512

    def forward(self, x, w):
        style = self.dense(w)
        style = style.view(-1, 2, self.channels, 1, 1)
        return x * (style[:, 0] + 1) + style[:, 1]


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.

    NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number, which is not trainable in
    this layer. However, the bias variable is trainable in this layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        gain=np.sqrt(2.0),
        lr_multiplier=1.0,
    ):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / np.sqrt(fan_in) * lr_multiplier
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.lr_multiplier = lr_multiplier

    def forward(self, x):
        return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier


class EpilogueBlock(nn.Module):
    """Implements the epilogue block of each conv block."""

    def __init__(
        self,
        resolution,
        channels,
        w_space_dim=512,
        normalization_fn="instance",
    ):
        super().__init__()
        self.apply_noise = NoiseApplyingLayer(resolution, channels)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if normalization_fn == "pixel":
            self.norm = PixelNormLayer()
        elif normalization_fn == "instance":
            self.norm = InstanceNormLayer()
        else:
            raise NotImplementedError(
                f"Not implemented normalization function: " f"{normalization_fn}!"
            )
        self.style_mod = StyleModulationLayer(channels, w_space_dim=w_space_dim)

    def forward(self, x, w):
        x = self.apply_noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.activate(x)
        x = self.norm(x)
        x = self.style_mod(x, w)
        return x


class FirstConvBlock(nn.Module):
    """Implements the first convolutional block.

    Basically, this block starts from a const input, which is
    `ones(channels, init_resolution, init_resolution)`.
    """

    def __init__(
        self, init_resolution, channels, w_space_dim=512):
        super().__init__()
        self.const = nn.Parameter(
            torch.ones(1, channels, init_resolution, init_resolution)
        )
        self.epilogue = EpilogueBlock(
            resolution=init_resolution,
            channels=channels,
            w_space_dim=w_space_dim,
        )

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        x = self.epilogue(x, w)
        return x


class UpConvBlock(nn.Module):
    """Implements the convolutional block with upsampling.
    """

    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        add_bias=False,
        wscale_gain=np.sqrt(2.0),
        wscale_lr_multiplier=1.0,
        w_space_dim=512,
    ):
        super().__init__()

        self.upsample = ResolutionScalingLayer()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=add_bias,
        )

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.blur = BlurLayer(channels=out_channels)
        self.epilogue = EpilogueBlock(
            resolution=resolution,
            channels=out_channels,
            w_space_dim=w_space_dim,
        )

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.conv(x) * self.scale
        x = self.blur(x)
        x = self.epilogue(x, w)
        return x


class UpConvBlockScale(nn.Module):
    """Implements the convolutional block with scale upsampling.
    """

    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        add_bias=False,
        wscale_gain=np.sqrt(2.0),
        wscale_lr_multiplier=1.0,
        w_space_dim=512,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(kernel_size, kernel_size, in_channels, out_channels)
        )
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.blur = BlurLayer(channels=out_channels)
        self.epilogue = EpilogueBlock(
            resolution=resolution,
            channels=out_channels,
            w_space_dim=w_space_dim,
        )

    def forward(self, x, w):
        kernel = self.weight * self.scale
        kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), "constant", 0.0)
        kernel = (
            kernel[1:, 1:] + kernel[:-1, 1:] + kernel[1:, :-1] + kernel[:-1, :-1]
        )
        kernel = kernel.permute(2, 3, 0, 1)
        x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
        x = self.blur(x)
        x = self.epilogue(x, w)
        return x


class ConvBlock(nn.Module):
    """Implements the normal convolutional block.

    Basically, this block is used as the second convolutional block for each
    resolution.
    """

    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        add_bias=False,
        wscale_gain=np.sqrt(2.0),
        wscale_lr_multiplier=1.0,
        w_space_dim=512,
    ):
        """Initializes the class with block settings.

        Args:
          resolution: Spatial resolution of current layer.
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          dilation: Dilation rate for convolution operation.
          add_bias: Whether to add bias onto the convolutional result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.
          w_space_dim: The dimension of disentangled latent space, w. This is used
            for style modulation.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=add_bias,
        )
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.epilogue = EpilogueBlock(
            resolution=resolution,
            channels=out_channels,
            w_space_dim=w_space_dim,
        )

    def forward(self, x, w):
        x = self.conv(x) * self.scale
        x = self.epilogue(x, w)
        return x


class LastConvBlock(nn.Module):
    """Implements the last convolutional block.

    Basically, this block converts the final feature map to RGB image.
    """

    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.scale = 1 / np.sqrt(in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv(x) * self.scale
        x = x + self.bias.view(1, -1, 1, 1)
        return x


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer, weight-scale layer,
    and activation layer in sequence.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        add_bias=False,
        wscale_gain=np.sqrt(2.0),
        wscale_lr_multiplier=0.01,
        activation_type="lrelu",
    ):
        """Initializes the class with block settings.

        Args:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels of the output tensor.
          add_bias: Whether to add bias onto the fully-connected result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.
          activation_type: Type of activation. Support `linear` and `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()

        self.fc = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=add_bias
        )
        self.wscale = WScaleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            gain=wscale_gain,
            lr_multiplier=wscale_lr_multiplier,
        )
        if activation_type == "linear":
            self.activate = nn.Identity()
        else:  # activation_type == "lrelu"
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.wscale(x)
        x = self.activate(x)
        return x
