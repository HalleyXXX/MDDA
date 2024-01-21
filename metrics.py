# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:44:51 2022

@author: GUI
"""

import torch

#iou
def iou_score(output, target, sigmoid=True, cut_off=0.5):
    smooth = 1e-5
    if torch.is_tensor(output):
        if sigmoid:
            output = torch.sigmoid(output).view(-1).data.cpu().numpy()
        else:
            output = output.view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(-1).data.cpu().numpy()
    output_ = output > cut_off
    target_ = target > cut_off
#    intersection = (output * target).sum()
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_coef(target, output, sigmoid=True, cut_off=0.5):
    smooth = 1e-5
    if sigmoid:
        output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    else:
        output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    target_ = target > cut_off
    output_ = output > cut_off
    intersection = (output_ & target_).sum()
    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)

#acc
def sensivity(target, output, sigmoid=True, cut_off=0.5):
    if torch.is_tensor(output):
        if sigmoid:
            outputs = torch.sigmoid(output).view(-1).data.cpu().numpy()
        else:
            outputs = output.data.cpu().numpy()
    if torch.is_tensor(target):
        targets = target.data.view(-1).cpu().numpy()
    outputs_ = outputs > cut_off
    targets_ = targets > cut_off
#    right = (outputs * targets).sum()
    right = (outputs_ & targets_ ).sum()
    num = targets_.sum()
#    num = outputs_.sum()
    return right/num