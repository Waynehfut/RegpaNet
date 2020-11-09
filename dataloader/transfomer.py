# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       transfomer.py
   Description :
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)


def train_transform(_config, p=1):
    return Compose([
        PadIfNeeded(min_height=_config['train_crop_height'], min_width=_config['train_crop_width'], p=1),
        RandomCrop(height=_config['train_crop_height'], width=_config['train_crop_width'], p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
    ], p=p)


def val_transform(_config, p=1):
    return Compose([
        PadIfNeeded(min_height=_config['val_crop_height'], min_width=_config['val_crop_width'], p=1),
        CenterCrop(height=_config['val_crop_height'], width=_config['val_crop_width'], p=1),
        Normalize(p=1)
    ], p=p)
