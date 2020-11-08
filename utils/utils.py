# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       utils.py
   Description :     Process utils function for project
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
import torch
import random


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
