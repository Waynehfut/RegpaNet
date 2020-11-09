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

from .common import CommFewShotDataset
from .endovis_prepare import binary_factor, parts_factor, instrument_factor
from albumentations.pytorch.functional import img_to_tensor


class EndoVisDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train',
                 problem_types=None):
        if problem_types is None:
            problem_types = ['binary', 'parts', 'instruments']
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_types = problem_types

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        binary_mask_name = self.problem_types[0]
        parts_mask_name = self.problem_types[1]
        instruments_mask_name = self.problem_types[2]
        image = load_image(img_file_name)
        binary_mask = load_mask(img_file_name, binary_mask_name)  # binary
        parts_mask = load_mask(img_file_name, parts_mask_name)  # parts
        instruments_mask = load_mask(img_file_name, instruments_mask_name)  # instruments
        data = {
            "image": image,
            binary_mask_name: binary_mask,
            parts_mask_name: parts_mask,
            instruments_mask_name: instruments_mask
        }
        augmented = self.transform(**data)
        image, binary_mask, parts_mask, instruments_mask = augmented["image"], augmented[binary_mask_name], augmented[
            parts_mask_name], augmented[instruments_mask_name]

        sample = {'image': img_to_tensor(image)}
        if self.mode == 'train':
            sample[binary_mask_name] = torch.from_numpy(np.expand_dims(binary_mask, 0)).float()
            sample[parts_mask_name] = torch.from_numpy(parts_mask).long()
            sample[instruments_mask_name] = torch.from_numpy(instruments_mask).long()

        else:
            sample[binary_mask_name] = []
            sample[parts_mask_name] = []
            sample[instruments_mask_name] = []
        return sample


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


def endo_fewshot_loader(file_names, shuffle=False, transform=None, problem_types=None,
                        n_ways=1, n_shots=1, n_queries=1,
                        batch_size=2, num_workers=1):
    if problem_types is None:
        problem_types = ['binary', 'parts', 'instruments']
    endo_dataset = EndoVisDataset(file_names=file_names, transform=transform, problem_types=problem_types)
    fewshot_dataset = CommFewShotDataset(datasets=endo_dataset, n_ways=n_ways, n_shots=n_shots, n_queries=n_queries)
    fewshot_loader = DataLoader(
        dataset=fewshot_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    return fewshot_loader
