# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       mytest.py
   Description :
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
from dataloader.endovis import EndoVisDataSet, endovis_simple_loader
from dataloader.endovis_prepare_train_val import get_split

train_file_names, val_file_names = get_split(0)

train_loader = endovis_simple_loader(train_file_names, shuffle=True, transform=None, problem_type='binary', batch_size=1)
val_loader=endovis_simple_loader(val_file_names, transform=None, problem_type='binary', batch_size=1)
print(len(train_loader),len(val_loader))