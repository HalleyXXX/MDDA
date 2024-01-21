# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:56:49 2023

@author: ROG
"""

import torch
from torch import nn
from .attention import Attention


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.stride = 1
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding),
                                  nn.GroupNorm(8,out_channel),
                                  nn.ReLU()
                                  )
    def forward(self, x):
        x = self.conv(x)
        return x

class Discriminator_feature(nn.Module):

    def __init__(self, input_channel=32):
        super().__init__()
        self.restored = False
        
        self.conv3x3 = nn.Sequential(ConvBlock(input_channel, 64, 3),
                                     ConvBlock(64, 128, 3),
                                     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
                                       )
        self.conv5x5 = nn.Sequential(ConvBlock(input_channel, 64, 5),
                                     ConvBlock(64, 128, 5),
                                     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
                                     )
        self.conv7x7 = nn.Sequential(ConvBlock(input_channel, 64, 7),
                                     ConvBlock(64, 128 ,7),
                                     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
                                     )
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.embedding3x3 = nn.Linear(256*256, 768)
        self.embedding5x5 = nn.Linear(256*256, 768)
        self.embedding7x7 = nn.Linear(256*256, 768)
        
        self.layer = nn.Sequential(nn.Linear(3*768, 768),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(768,768),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(768, 2),
                                   nn.LogSoftmax(dim=1)
                                   )

        self.attention = Attention(out_size=768)
        
    def forward(self, x):
        h_3x3 = self.conv3x3(x)
        h_5x5 = self.conv5x5(x)
        h_7x7 = self.conv7x7(x)
        
        h_3x3 = self.embedding3x3(h_3x3.view(x.size(0), -1))
        h_5x5 = self.embedding5x5(h_5x5.view(x.size(0), -1))
        h_7x7 = self.embedding7x7(h_7x7.view(x.size(0), -1))
                        
        h_3x3 = torch.unsqueeze(h_3x3, dim=1)
        h_5x5 = torch.unsqueeze(h_5x5, dim=1)
        h_7x7 = torch.unsqueeze(h_7x7, dim=1)
        h = self.attention(torch.cat([h_3x3,h_5x5,h_7x7],dim=1))
        h = h.view(h.size(0), -1)
        h = self.layer(h)        
        
        return h

class Discriminator_edge(nn.Module):

    def __init__(self, dim=500):
        super().__init__()
        self.restored = False
        
        self.layer = nn.Sequential(
                nn.Linear(256*256, 500),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(500, 500),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(500, 2),
                nn.LogSoftmax(dim=1)
                )
        
    def forward(self, x):
        
        h = self.layer(x)        
        
        return h

