# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:07:02 2022

@author: GUI
"""

import torch
from torch import nn
from .attention import se_block, cbam_block, eca_block

attention_blok = [se_block, cbam_block, eca_block]

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UnetppEncoder(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, attention=False, attention_model = 0, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.attention = attention
        self.restored = False

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.attention:
            if self.deep_supervision:
                self.feat1_attention = attention_blok[attention_model](nb_filter[0])
                self.feat2_attention = attention_blok[attention_model](nb_filter[0])
                self.feat3_attention = attention_blok[attention_model](nb_filter[0])
                self.feat4_attention = attention_blok[attention_model](nb_filter[0])
            else:
                self.feat_attention = attention_blok[attention_model](nb_filter[0])
            
        

    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            if self.attention:
                x0_1 = self.feat1_attention(x0_1)
                x0_2 = self.feat2_attention(x0_2)
                x0_3 = self.feat3_attention(x0_3)
                x0_4 = self.feat4_attention(x0_4)
            return [x0_1, x0_2, x0_3, x0_4]
        else:
            if self.attention:
                x0_4 = self.feat_attention(x0_4)
            return x0_4


       
class UnetppClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.restored = False
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    def forward(self, input, only_output4=False):
        if self.deep_supervision:
            if only_output4:
                output4 = self.final4(input)
                return output4
            else:
                output1 = self.final1(input[0])
                output2 = self.final2(input[1])
                output3 = self.final3(input[2])
                output4 = self.final4(input[3])
                return [output1, output2, output3, output4]

        else:
            output = self.final(input)
            return output