# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:37:19 2022

@author: GUI
"""

import os
import torch
import albumentations as transforms
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf

import params
from dataset import Dataset

def load_src_data(kind='train'):
    img_ids = glob(os.path.join(params.src_data_path, kind, 'images', '*' + params.img_ext))
#    print(os.path.join(params.src_data_path, kind, 'images', '*' + params.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
#    train_img_ids = 4*train_img_ids
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
    val_transform = Compose([
        transforms.Resize(params.input_h, params.input_w),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(params.src_data_path,kind, 'images'),
        mask_dir=os.path.join(params.src_data_path,kind, 'masks'),
        img_ext=params.img_ext,
        mask_ext=params.mask_ext,
        num_classes=params.num_classes,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(params.src_data_path,kind, 'images'),
        mask_dir=os.path.join(params.src_data_path,kind, 'masks'),
        img_ext=params.img_ext,
        mask_ext=params.mask_ext,
        num_classes=params.num_classes,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        drop_last=True)#不能整除的batch是否就不要了
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        drop_last=False)
    
    return train_loader, val_loader