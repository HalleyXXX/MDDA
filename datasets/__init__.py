# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:06:51 2022

@author: GUI
"""

from .load_src_data import load_src_data as get_H_data
from .load_tgt_data import load_tgt_data as get_L_data
from .load_data_test import load_data_test as get_data_test

__all__ = (get_H_data, get_L_data, get_data_test)