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
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from data import grid_image
from model import model_device, model_setenv, get_encoder, get_decoder

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", type=str, default="images/*.png", help="input image")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_setenv()
    device = model_device()

    encoder_model = get_encoder("models/stylegan2_encoder.pth")
    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    decoder_model = get_decoder("models/stylegan2_decoder.pth")
    decoder_model = decoder_model.to(device)
    decoder_model.eval()


    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB").resize((256, 256))
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            wcode = encoder_model(input_tensor)
            output_tensor = decoder_model(wcode).clamp(0, 1.0)

        image = grid_image([input_tensor, output_tensor], nrow=2)
        image.save("{}/image_{:02d}.jpg".format(args.output, index + 1))