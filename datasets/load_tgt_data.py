# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:11:05 2022

@author: GUI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:37:19 2022

@author: GUI
"""

import os
import torch
import numpy as np
import albumentations as transforms
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
import random
import params
from dataset import Dataset

def load_tgt_data(path, num, kind, sample_num=None):
    if not num == None:
        img_ids = glob(os.path.join(path, kind, 'images_' + num, '*' + params.img_ext))
    else:
        img_ids = glob(os.path.join(path, kind, 'images', '*' + params.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    if sample_num != None:
        np.random.seed(1001)
        img_ids = random.sample(img_ids, sample_num)
#    temp = []
#    for i in range(30):
#        temp += img_ids.copy()
#    img_ids = temp

    #数据增强：
    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),#按照归一化的概率选择执行哪一个
        transforms.Resize(params.input_h, params.input_w),
        transforms.Normalize(),
    ])

    if not num == None:
        img_dir = os.path.join(path, kind, 'images_' + num)
    else:
        img_dir = os.path.join(path, kind, 'images')
    train_dataset = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir='',
        img_ext=params.img_ext,
        mask_ext=params.mask_ext,
        num_classes=params.num_classes,
        transform=train_transform,
        mask_flag = False,
#        max_iters = params.max_iters,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        drop_last=True)#不能整除的batch是否就不要了

    return train_loader



