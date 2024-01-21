# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:45:20 2022

@author: GUI
"""

import os

epochs = 200
batch_size = 2

name = '001'

#model
input_channels = 3
num_classes = 1
input_w = 256
input_h = 256
deep_supervision = True
deep_layer = 3 #(0,1,2,3)\
##target_model
t_scheduler = True
t_gamma = 0.1
d_scheduler = True
d_gama = 0.1

#params for source dataset
src_model = 'unetpp'
src_dataset = 'bone_H_'+str(input_w)
src_data_path = r'inputs\bone_H_'+str(input_w)
src_encoder_restore = os.path.join('snapshots', name,'source-encoder-final.pth')
src_classifier_restore = os.path.join('snapshots', name,'source-classifier-final.pth')

src_model_trained = True
img_ext = '.png'
mask_ext = '.png'
loss = 'BCEDiceLoss'

#params for target dataset
tgt_dataset = 'bone_L_'+str(input_w)
tgt_data_path = r'inputs\bone_L_'+str(input_w)
tgt_encoder_restore = os.path.join('snapshots',name,'target-encoder-10.pth')
tgt_model_trained = True

#optimizer for pretrain
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
nesterov = False

#optimizer for adapt
d_learning_rate = 1e-6
c_learning_rate = 1e-6
beta1 = 0.5
beta2 = 0.9

#scheduler
min_lr = 1e-7

num_workers = 0
save_step = 10