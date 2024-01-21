# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:01:01 2022

@author: GUI
"""

import params
import torch
from utils import get_data_loader,init_model
from models import UnetppEncoder, UnetppClassifier, Discriminator_feature, Discriminator_edge
from core import train_src, train_tgt, test

if __name__ =='__main__':
    #load dataset
    src_data_loader, src_data_loader_val = get_data_loader(params.src_dataset, kind='train')
    src_data_loader_test = get_data_loader(params.src_dataset, test_flag=True, kind='test')
    
    tgt_data_loader = get_data_loader(params.tgt_dataset, kind='train', sample_num=None)
    tgt_data_loader_val = get_data_loader(params.tgt_dataset, test_flag=True, kind='val')
    tgt_data_loader_test = get_data_loader(params.tgt_dataset, test_flag=True, kind='test')

    #load source models
    src_encoder = init_model(net=UnetppEncoder(num_classes=params.num_classes, input_channels=params.input_channels, deep_supervision=params.deep_supervision),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=UnetppClassifier(num_classes=params.num_classes, input_channels=params.input_channels, deep_supervision=params.deep_supervision),
                                restore=params.src_classifier_restore)

    src_reload = False
    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
                src_encoder, src_classifier, src_data_loader, src_data_loader_val)
        src_reload = True
    if src_reload:
        print('>Reload the best src_encoder from %s<'%(params.src_encoder_restore))
        src_encoder.load_state_dict(torch.load(params.src_encoder_restore))
        print('>Reload the best src_classifier from %s<'%(params.src_classifier_restore))
        src_classifier.load_state_dict(torch.load(params.src_classifier_restore))
        
    # test source model
    print("=== Evaluating classifier for source domain ===")
    test(src_encoder, src_classifier, src_data_loader_test)
    
    #load target models
    tgt_encoder = init_model(net=UnetppEncoder(num_classes=params.num_classes, input_channels=params.input_channels, deep_supervision=params.deep_supervision,
                                               attention=params.t_attention,attention_model=params.t_attention_model),
            restore=params.tgt_encoder_restore)
    
    critic_feature = init_model(net=Discriminator_feature())
    critic_edge = init_model(net=Discriminator_edge())
    
    # init weights of target encoder with those of source encoder
    tgt_reload = False
    if not tgt_encoder.restored:
            tgt_encoder.load_state_dict(src_encoder.state_dict(), strict=False)
    
    if not (tgt_encoder.restored and params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic_feature, critic_edge,
                                            src_data_loader, tgt_data_loader,
                                            src_classifier,tgt_data_loader_val)

        tgt_reload = True
    if tgt_reload:
        print('>Reload the best tgt_encoder from %s<'%(params.tgt_encoder_restore))
        tgt_encoder.load_state_dict(torch.load(params.tgt_encoder_restore))
        
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    test(src_encoder, src_classifier, tgt_data_loader_test)
    print(">>> domain adaption <<<")
    test(tgt_encoder, src_classifier, tgt_data_loader)
    print('>val results<')
    test(tgt_encoder, src_classifier, tgt_data_loader_val)
    print('>test results<')
    test(tgt_encoder, src_classifier, tgt_data_loader_test)