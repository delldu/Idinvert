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
from model import model_device, model_setenv, get_encoder, get_decoder, get_refiner
from stylegan2_refiner import Attribute

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
    model = get_refiner()
    model = model.to(device)
    model.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total=len(image_filenames))

    attribute = Attribute()

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB").resize((256, 256))
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            wcode = model.encoder(input_tensor)
            output_tensor1 = model.decoder(wcode)

        # require grad
        refine_wcode = model(input_tensor)

        with torch.no_grad():
            output_tensor2 = model.decoder(refine_wcode)


        for name in ['age', 'gender', 'pose', 'express', 'glass']:
            list_ext = [input_tensor, output_tensor1, output_tensor2]
            print("Start length(list_ext) == ", len(list_ext))

            modified_wcodes = attribute.get_wcodes(refine_wcode, name)
            print(modified_wcodes.size())

            for j in range(modified_wcodes.size(1)):
                if (j != 3):
                    with torch.no_grad():
                        output_tensor = model.decoder(modified_wcodes[:, j])
                    list_ext.append(output_tensor)
            print("Stop length(list_ext) == ", len(list_ext))

            image = grid_image(list_ext, nrow=3)
            image.save("{}/image_{:03d}_{}.png".format(args.output, index + 1, name))
