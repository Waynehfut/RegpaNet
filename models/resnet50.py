# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       resnet50.py
   Description :
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        return self.features(x)
