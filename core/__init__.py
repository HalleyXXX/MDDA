# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:55:07 2022

@author: GUI
"""

from .pretrain import train_src
from .adapt import train_tgt
from .test import test

__all__ = (train_src, train_tgt, test)