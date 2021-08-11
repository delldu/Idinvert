"""Convert attributes."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 10日 星期二 15:49:52 CST
# ***
# ************************************************************************************/
#

import torch
import numpy as np
import pdb

age = np.load("age.npy", allow_pickle=True)[()]
express = np.load("expression.npy", allow_pickle=True)[()]
glass = np.load("eyeglasses.npy", allow_pickle=True)[()]
gender = np.load("gender.npy", allow_pickle=True)[()]
pose = np.load("pose.npy", allow_pickle=True)[()]


stylegan_feature = {}

# age
layers = np.array(age['meta_data']['manipulate_layers'])
stylegan_feature['age.layer'] = torch.from_numpy(layers)
stylegan_feature['age.attrs'] = torch.from_numpy(age['boundary'])


# express
layers = np.array(express['meta_data']['manipulate_layers'])
stylegan_feature['express.layer'] = torch.from_numpy(layers)
stylegan_feature['express.attrs'] = torch.from_numpy(express['boundary'])


# glass
layers = np.array(glass['meta_data']['manipulate_layers'])
stylegan_feature['glass.layer'] = torch.from_numpy(layers)
stylegan_feature['glass.attrs'] = torch.from_numpy(glass['boundary'])


# gender
layers = np.array(gender['meta_data']['manipulate_layers'])
stylegan_feature['gender.layer'] = torch.from_numpy(layers)
stylegan_feature['gender.attrs'] = torch.from_numpy(gender['boundary'])


# pose
layers = np.array(pose['meta_data']['manipulate_layers'])
stylegan_feature['pose.layer'] = torch.from_numpy(layers)
stylegan_feature['pose.attrs'] = torch.from_numpy(pose['boundary'])


torch.save(stylegan_feature, "stylegan_feature.pth")

pdb.set_trace()
