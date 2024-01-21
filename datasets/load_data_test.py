# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:40:02 2022

@author: GUI
"""

import os
import torch
import albumentations as transforms
from glob import glob
from albumentations.core.composition import Compose

import params
from dataset import Dataset

def load_data_test(path, kind):
    test_img_ids = glob(os.path.join(path, kind, 'images', '*'+ params.img_ext))
    for i in range(len(test_img_ids)):
        test_img_ids[i]=os.path.splitext(os.path.basename(test_img_ids[i]))[0]

    test_transform = Compose([
        transforms.Resize(params.input_h, params.input_w),
        transforms.Normalize(),
    ])
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(path, kind, 'images'),
        mask_dir=os.path.join(path, kind, 'masks'),
        img_ext=params.img_ext,
        mask_ext=params.mask_ext,
        num_classes=params.num_classes,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        drop_last=False)
    
    return test_loader
