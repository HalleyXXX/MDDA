# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:32:38 2022

@author: GUI
"""

import os
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.optim import lr_scheduler

import params
import losses
from utils import AverageMeter
from metrics import iou_score, dice_coef, accuracy

def train_src(encoder, classifier, train_loader, val_loader):
    
    os.makedirs('outputs/%s' % params.name, exist_ok=True)
    os.makedirs('snapshots/%s' % params.name, exist_ok=True)
    
    if torch.cuda.is_available():
        criterion = losses.__dict__[params.loss]().cuda()
    else:
        criterion = losses.__dict__[params.loss]()
    
    params_encoder = filter(lambda p: p.requires_grad, encoder.parameters())
    params_classifier = filter(lambda p: p.requires_grad, classifier.parameters())
    optimizer = optim.SGD(list(params_encoder)+list(params_classifier), lr=params.lr, momentum=params.momentum,
                              nesterov=params.nesterov, weight_decay=params.weight_decay)
    
    scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.epochs, eta_min=params.min_lr)
    
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('acc',[]),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice',[]),
        ('val_acc',[])
        
    ])
    best_iou = 0
    acc_ = 0
    trigger = 0
    for epoch in range(params.epochs):
        print('Epoch [%d/%d]' % (epoch+1, params.epochs))

        # train for one epoch
        train_log = train(train_loader, encoder, classifier, criterion, optimizer)

        # evaluate on validation set
        val_log = validate(val_loader, encoder, classifier, criterion)
        scheduler.step()
        print('loss %.4f - iou %.4f - dice%.4f - acc%.4f\n - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_acc%.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], train_log['acc'], 
              val_log['loss'], val_log['iou'], val_log['dice'], val_log['acc']))
        print('\n==> best_iou = %.4f, best_acc = %.4f'%(best_iou, acc_))

        log['epoch'].append(epoch)
        log['lr'].append(params.lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['acc'].append(train_log['acc'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_acc'].append(val_log['acc'])

        pd.DataFrame(log).to_csv('snapshots/%s/log_pretrain.csv' %
                                 params.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(encoder.state_dict(), 'snapshots/%s/source-encoder-final.pth' %
                       params.name)
            torch.save(classifier.state_dict(), 'snapshots/%s/source-classifier-final.pth' %
                       params.name)
            best_iou = val_log['iou']
            acc_ = val_log['acc']
            print("=> saved best model")
            trigger = 0
        elif (val_log['iou'] == best_iou) and (val_log['acc'] > acc_):
            torch.save(encoder.state_dict(), 'snapshots/%s/source-encoder-final.pth' %
                       params.name)
            torch.save(classifier.state_dict(), 'snapshots/%s/source-classifier-final.pth' %
                       params.name)
            best_iou = val_log['iou']
            acc_ = val_log['acc']
            print("=> saved best model")
            trigger = 0
        # early stopping
        if params.early_stopping >= 0 and trigger >= params.early_stopping:
            print("=> early stopping")
            break
        
        torch.cuda.empty_cache()
        
    return encoder, classifier

def train(train_loader, encoder, classifier, criterion, optimizer):

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'acc': AverageMeter(),
                 }
    
    encoder.train()
    classifier.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if params.deep_supervision:
            outputs = classifier(encoder(input))
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(target, outputs[-1])
            acc = accuracy(target, outputs[-1])
            
        else:
            output = classifier(encoder(input))
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(target , output)
            acc = acc(target, output)
            
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['acc'].update(acc, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('acc',avg_meters['acc'].avg)
            
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg),
                        ('acc',avg_meters['acc'].avg)
                        ])


def validate(val_loader, encoder, classifier, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'acc': AverageMeter()}

    # switch to evaluate mode
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if params.deep_supervision:
                outputs = classifier(encoder(input))
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(target, outputs[-1])
                acc = accuracy(target,outputs[-1])
            else:
                output = classifier(encoder(input))
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(target, output )
                acc=accuracy(target, output)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('acc', avg_meters['acc'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('acc',avg_meters['acc'].avg)
                        ])