# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       endovis.py
   Description :
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .endovis_prepare import binary_factor, parts_factor, instrument_factor
from albumentations.pytorch.functional import img_to_tensor


def get_split(fold, data_path):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = Path(data_path) / 'CroppedTrain'

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names


def endovis_simple_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1,
                          num_workers=0):
    return DataLoader(
        dataset=EndoVisDataSet(file_names, transform=transform, problem_type=problem_type),
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )


def endovis_few_shot_loader(train_file_names, val_file_names, problem_type, n_ways, n_shots, n_queries, batch_size):
    support_loader = endovis_simple_loader(train_file_names, shuffle=True, transform=None,
                                           problem_type=problem_type,
                                           batch_size=n_shots)
    query_loader = endovis_simple_loader(val_file_names, transform=None, problem_type=problem_type,
                                         batch_size=n_queries)
    # TODO handle the batch size in few shot
    print('ok')


class EndoVisDataSet(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    mask_folder = ''
    factor = 0
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)
