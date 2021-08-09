# python 3.7
"""Contains the generator class of StyleGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator
from .stylegan_generator_network import StyleGANGeneratorNet
import pdb

__all__ = ["StyleGANGenerator"]


class StyleGANGenerator(BaseGenerator):
    """Defines the generator class of StyleGAN.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, logger=None):
        self.gan_type = "stylegan"
        super().__init__(model_name, logger)
        self.lod = self.net.synthesis.lod.to(self.cpu_device).tolist()
        self.logger.info(f"Current `lod` is {self.lod}.")

    def build(self):
        self.z_space_dim = getattr(self, "z_space_dim", 512)
        self.w_space_dim = getattr(self, "w_space_dim", 512)
        self.num_mapping_layers = getattr(self, "num_mapping_layers", 8)
        self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
        self.truncation_layers = 8
        self.net = StyleGANGeneratorNet(
            resolution=self.resolution,
            z_space_dim=self.z_space_dim,
            w_space_dim=self.w_space_dim,
            num_mapping_layers=self.num_mapping_layers,
            image_channels=self.image_channels,
            truncation_psi=self.truncation_psi,
            truncation_layers=self.truncation_layers,
        )
        self.num_layers = self.net.num_layers
        self.model_specific_vars = ["truncation.truncation"]

    def sample(self, num, latent_space_type="z", **kwargs):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        print("sample: latent_space_type == ", latent_space_type)

        latent_space_type = latent_space_type.lower()
        if latent_space_type == "z":
            latent_codes = np.random.randn(num, self.z_space_dim)
        elif latent_space_type in ["w", "wp"]:
            z = self.easy_sample(num, latent_space_type="z")
            latent_codes = []
            for inputs in self.get_batch_inputs(z, self.ram_size):
                outputs = self.easy_synthesize(
                    latent_codes=inputs,
                    latent_space_type="z",
                    generate_style=False,
                    generate_image=False,
                )
                latent_codes.append(outputs[latent_space_type])
            latent_codes = np.concatenate(latent_codes, axis=0)
            if latent_space_type == "w":
                assert latent_codes.shape == (num, self.w_space_dim)
            elif latent_space_type == "wp":
                assert latent_codes.shape == (num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f"Latent space type `{latent_space_type}` is invalid!")

        return latent_codes.astype(np.float32)

    def preprocess(self, latent_codes, latent_space_type="z", **kwargs):
        latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
        return latent_codes.astype(np.float32)

    def _synthesize(
        self,
        latent_codes,
        latent_space_type="z",
        labels=None,
        generate_style=False,
        generate_image=True,
    ):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)
          labels: Additional labels for conditional generation.
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        # latent_codes.shape -- (2, 14, 512)
        # latent_space_type = 'wp'
        # labels = None
        # generate_style = False
        # generate_image = True

        latent_space_type = latent_space_type.lower()
        results = {}

        # latent_space_type == "wp"
        if latent_codes.ndim != 3 or latent_codes.shape[1:] != (
            self.num_layers,
            self.w_space_dim,
        ):
            raise ValueError(
                f"Latent codes should be with shape [batch_size, "
                f"num_layers, w_space_dim], where `num_layers` equals "
                f"to {self.num_layers}, and `w_space_dim` equals to "
                f"{self.w_space_dim}!\n"
                f"But {latent_codes.shape} is received!"
            )
        wps = self.to_tensor(latent_codes.astype(np.float32))
        results["wp"] = latent_codes

        # pp wps.size() -- torch.Size([2, 14, 512])

        # generate_style -- False
        # if generate_style:
        #     for i in range(self.num_layers):
        #         style = self.net.synthesis.__getattr__(
        #             f"layer{i}"
        #         ).epilogue.style_mod.dense(wps[:, i, :])
        #         results[f"style{i:02d}"] = self.get_value(style)
        # generate_image == True
        if generate_image:
            images = self.net.synthesis(wps)
            results["image"] = self.get_value(images)

        if self.use_cuda:
            torch.cuda.empty_cache()

        return results

    def synthesize(
        self,
        latent_codes,
        latent_space_type="z",
        labels=None,
        generate_style=False,
        generate_image=True,
        **kwargs,
    ):
        return self.batch_run(
            latent_codes,
            lambda x: self._synthesize(
                x,
                latent_space_type=latent_space_type,
                labels=labels,
                generate_style=generate_style,
                generate_image=generate_image,
            ),
        )
