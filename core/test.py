# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:09:56 2022

@author: GUI
"""

import torch
from tqdm import tqdm

import params
from metrics import iou_score, dice_coef, sensitivity

import sys
sys.path.append('..')
from utils import AverageMeter

def test(encoder, classifier, data_loader, return_flag=False):
    
    encoder.eval()
    classifier.eval()
    
    avg_meter = {'iou':AverageMeter(),
                 'dice':AverageMeter(),
                 'sensitivity':AverageMeter()
                 }

    with torch.no_grad():
        for input, target, meta in tqdm(data_loader, total=len(data_loader)):
            
            input = input.cuda()
            target = target.cuda()

            # compute output
            if params.deep_supervision:
                deep_layer = params.deep_layer
                output = classifier(encoder(input))[deep_layer]
            else:
                output = classifier(encoder(input))

            iou = iou_score(output, target)
            dice = dice_coef(target, output)
            sen = sensitivity(target, output)
            avg_meter['iou'].update(iou, input.size(0))
            avg_meter['dice'].update(dice, input.size(0))
            avg_meter['sensitivity'].update(sen, input.size(0))
            
    print('IoU: %.4f - dice: %.4f - sensitivity: %.4f' %(avg_meter['iou'].avg,avg_meter['dice'].avg,avg_meter['sensitivity'].avg))
    torch.cuda.empty_cache()
    if return_flag:
        return avg_meter['iou'].avg, avg_meter['sensitivity'].avg