# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:19:45 2022

@author: GUI
"""

"""Adversarial adaptation to train target encoder."""

import os

import torch
from torch import nn
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.autograd import Variable

import params
from utils import make_variable, AverageMeter, set_requires_grad
from core.test import test


def nn_conv2d_Laplace(im):
    
    conv_op = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = conv_op(Variable(im))
    
    return edge_detect

def train_tgt(src_encoder, tgt_encoder, critic_feature, critic_edge,
              src_data_loader, tgt_data_loader,
              src_classifier,tgt_data_loader_val):
    
    os.makedirs(os.path.join('snapshots', params.name), exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate)
    optimizer_critic = optim.Adam(list(critic_feature.parameters())+list(critic_edge.parameters()),
                                  lr=params.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    if params.t_scheduler:
        t_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_tgt, T_max=params.epochs, eta_min=params.min_lr)
    if params.d_scheduler:
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_tgt, T_max=params.epochs, eta_min=params.min_lr)
    
    #set data log
    log = OrderedDict([
            ('epoch', []),
            ('d_loss', []),
            ('g_loss', []),
            ])
    best_d_loss = 999
    
    for epoch in range(params.epochs):
        print('Epoch [%d/%d]' % (epoch+1, params.epochs))
        train_log = train(src_encoder, tgt_encoder, critic_feature, src_classifier, critic_edge, 
              src_data_loader, tgt_data_loader,
              criterion, optimizer_tgt, optimizer_critic,
              len_data_loader)
        print('d_loss %.4f - g_loss %.4f - d_acc %.4f'
              % (train_log['d_loss'], train_log['g_loss'], train_log['acc']))
        
        log['epoch'].append(epoch)
        log['d_loss'].append(train_log['d_loss'])
        log['g_loss'].append(train_log['g_loss'])
        log['acc'].append(train_log['acc'])
        
        pd.DataFrame(log).to_csv('snapshots/%s/log_adapt.csv' %
                                 params.name, index=False)
        if params.t_scheduler:
            t_scheduler.step()
        if params.d_scheduler:
            d_scheduler.step()
        if train_log['d_loss'] < best_d_loss:
            best_d_loss = train_log['d_loss']
        if epoch % params.save_step == 0:
            torch.save(tgt_encoder.state_dict(), os.path.join(
                    'snapshots', params.name, 'target-encoder-'+str(epoch)+'.pth'))

        torch.cuda.empty_cache()
        
    torch.save(tgt_encoder.state_dict(), os.path.join('snapshots', params.name, 'target-encoder-final.pth'))

    return tgt_encoder


def train(src_encoder, tgt_encoder, critic_feature, src_classifier, critic_edge, 
              src_data_loader, tgt_data_loader,
              criterion, optimizer_tgt, optimizer_critic,
              len_data_loader):
    
    avg_meters = {'d_loss': AverageMeter(),
                      'g_loss': AverageMeter(),
                      'acc': AverageMeter()
                }
    
    tgt_encoder.train()
    critic_feature.train()
    critic_edge.train()
    
    pbar = tqdm(total=len_data_loader)
    
    # zip source and target data pair
    data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
    for step, ((images_src, _, _), (images_tgt, _)) in data_zip:
        
        ###########################
        # 2.1 train discriminator #
        ###########################
        # make images variable
        set_requires_grad(tgt_encoder, requires_grad=False)
        set_requires_grad(critic_feature, requires_grad=True)
        set_requires_grad(critic_edge, requires_grad=True)
        
        tgt_encoder.eval()
        critic_feature.train()
        critic_edge.train()
        
        images_src = make_variable(images_src)
        images_tgt = make_variable(images_tgt)

        optimizer_critic.zero_grad()

        if params.deep_supervision:
            deep_layer = params.deep_layer
            feat_src = src_encoder(images_src)[deep_layer]
            feat_tgt = tgt_encoder(images_tgt)[deep_layer]
        else:
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
        
        feat_concat = torch.cat((feat_src, feat_tgt), 0)
        pred_concat = critic_feature(feat_concat.detach())

        label_src = make_variable(torch.ones(feat_src.size(0)).long())
        label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
        label_concat = torch.cat((label_src, label_tgt), 0)
        
        #sobel
        pred_src = src_classifier(src_encoder(images_src))[deep_layer]
        pred_tgt = src_classifier(tgt_encoder(images_tgt))[deep_layer]
        
        edge_src = nn_conv2d_Laplace(pred_src)
        edge_tgt = nn_conv2d_Laplace(pred_tgt)
        feat_edge_concat = torch.cat((edge_src, edge_tgt), 0)
        pred_concat_feature_sobel = critic_edge(feat_edge_concat.detach().view(feat_edge_concat.size(0), -1))
        
        loss_critic = 0.01*criterion(pred_concat, label_concat) + 1*criterion(pred_concat_feature_sobel, label_concat)
        
        loss_critic.backward()
        optimizer_critic.step()

        pred_cls = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_cls == label_concat).float().mean()

        ############################
        # 2.2 train target encoder #
        ############################
        layers = [tgt_encoder.conv0_4]
        set_requires_grad(tgt_encoder, requires_grad=True, layers=layers)
        set_requires_grad(critic_feature, requires_grad=False)
        set_requires_grad(critic_edge, requires_grad=False)
            
        tgt_encoder.train()
        critic_feature.eval()
        critic_edge.eval()
        
        # zero gradients for optimizer
        optimizer_critic.zero_grad()
        optimizer_tgt.zero_grad()

        # extract and target features
        if params.deep_supervision:
            feat_tgt = tgt_encoder(images_tgt)[deep_layer]
        else:
            feat_tgt = tgt_encoder(images_tgt)

        pred_tgt_feature = critic_feature(feat_tgt)

        pred_tgt = src_classifier(tgt_encoder(images_tgt))[deep_layer]
        edge_tgt = nn_conv2d_Laplace(pred_tgt)

        pred_feature_sobel = critic_edge(edge_tgt.view(edge_tgt.size(0), -1))

        loss_tgt = 0.01*criterion(pred_tgt_feature, label_tgt) + 1*criterion(pred_feature_sobel, label_tgt)
        loss_tgt.backward()

        optimizer_tgt.step()
        
        avg_meters['d_loss'].update(loss_critic.item(), images_src.size(0))
        avg_meters['g_loss'].update(loss_tgt.item(), images_src.size(0))
        avg_meters['acc'].update(acc.item(), images_src.size(0))

        postfix = OrderedDict([
                ('d_loss', avg_meters['d_loss'].avg),
                ('g_loss', avg_meters['g_loss'].avg),
                ('acc', avg_meters['acc'].avg),
            ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    
    return OrderedDict([('d_loss', avg_meters['d_loss'].avg),
                        ('g_loss', avg_meters['g_loss'].avg),
                        ('acc', avg_meters['acc'].avg),
                        ])