"""Onnx Model Tools."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 10日 星期二 15:49:52 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import pdb  # For debug
import time

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Import Model Method
# ***
# ************************************************************************************/
#
from model import get_encoder, get_decoder, get_vgg16


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx Model Engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


def onnx_forward(onnx_model, input):
    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


if __name__ == "__main__":
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--export", help="export onnx model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify onnx model", action="store_true")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #

    encoder_input = torch.randn(1, 3, 256, 256)
    decoder_input = torch.randn(1, 1, 14, 512)
    vgg16_input = torch.randn(1, 3, 256, 256)

    encoder_checkpoint = "models/stylegan2_encoder.pth"
    decoder_checkpoint = "models/stylegan2_decoder.pth"
    vgg16_checkpoint = "models/stylegan2_vgg16.pth"

    encoder_onnx_file_name = "{}/stylegan2_encoder.onnx".format(args.output)
    decoder_onnx_file_name = "{}/stylegan2_decoder.onnx".format(args.output)
    vgg16_onnx_file_name = "{}/stylegan2_vgg16.onnx".format(args.output)

    def export_encoder_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_encoder(encoder_checkpoint)
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(encoder_onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        # dynamic_axes = {'input': {0: "batch"},'output': {0: "batch"}}

        torch.onnx.export(
            torch_model,
            encoder_input,
            encoder_onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            export_params=True,
        )

        # 3. Optimize model
        print("Checking model ...")
        onnx_model = onnx.load(encoder_onnx_file_name)
        onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/stylegan_encoder.onnx')"

    def export_decoder_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_decoder(decoder_checkpoint)
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(decoder_onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        # dynamic_axes = {'input': {0: "batch"},'output': {0: "batch"}}

        torch.onnx.export(
            torch_model,
            decoder_input,
            decoder_onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            export_params=True,
        )

        # 3. Optimize model
        print("Checking model ...")
        onnx_model = onnx.load(decoder_onnx_file_name)
        onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/stylegan2_decoder.onnx')"

    def export_vgg16_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_vgg16(vgg16_checkpoint)
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(vgg16_onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        # dynamic_axes = {'input': {0: "batch"},'output': {0: "batch"}}

        torch.onnx.export(
            torch_model,
            vgg16_input,
            vgg16_onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            export_params=True,
        )

        # 3. Optimize model
        print("Checking model ...")
        onnx_model = onnx.load(vgg16_onnx_file_name)
        onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/stylegan2_vgg16.onnx')"


    def verify_encoder_onnx():
        """Verify onnx model."""

        torch_model = get_encoder(encoder_checkpoint)
        torch_model.eval()

        onnxruntime_engine = onnx_load(encoder_onnx_file_name)

        with torch.no_grad():
            torch_output = torch_model(encoder_input)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(encoder_input)
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Onnx model {} tested with ONNXRuntime, result sounds good !".format(
                encoder_onnx_file_name
            )
        )

    def verify_decoder_onnx():
        """Verify onnx model."""

        torch_model = get_decoder(decoder_checkpoint)
        torch_model.eval()

        onnxruntime_engine = onnx_load(decoder_onnx_file_name)

        with torch.no_grad():
            torch_output = torch_model(decoder_input)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(decoder_input)
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Onnx model {} tested with ONNXRuntime, result sounds good !".format(
                decoder_onnx_file_name
            )
        )


    def verify_vgg16_onnx():
        """Verify onnx model."""

        torch_model = get_vgg16(vgg16_checkpoint)
        torch_model.eval()

        onnxruntime_engine = onnx_load(vgg16_onnx_file_name)

        with torch.no_grad():
            torch_output = torch_model(vgg16_input)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(vgg16_input)
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Onnx model {} tested with ONNXRuntime, result sounds good !".format(
                vgg16_onnx_file_name
            )
        )


    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    if args.export:
        export_encoder_onnx()
        export_decoder_onnx()
        export_vgg16_onnx()

    if args.verify:
        verify_encoder_onnx()
        verify_decoder_onnx()
        verify_vgg16_onnx()

