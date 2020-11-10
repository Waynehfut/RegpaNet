# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       common.py
   Description :
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""
import random

from torch.utils.data import Dataset


class CommFewShotDataset(Dataset):
    def __init__(self, datasets, n_ways, n_shots, n_queries):
        """
        Transfer simple dataset into N_ways N_shots datasets, with N_queries.

        :param datasets: Simple dataset
        :param n_ways: Tasks, normally will be 1, for EndoVis can be 1,2,3
        :param n_shots: Train sample in a way
        :param n_queries: Test sample in a way
        """
        self.datasets = datasets
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        if len(self.datasets[0]) < (self.n_ways + 1):
            raise ValueError(f'Dataset not support the way larger than {self.n_ways} ways')

    def __len__(self):
        return int(len(self.datasets) / (self.n_shots + self.n_queries))

    def __getitem__(self, idx):
        step = self.n_shots + self.n_queries
        support_images = []
        support_binary = []
        support_parts = []
        support_instruments = []
        query_images = []
        query_binary = []
        query_parts = []
        query_instruments = []
        for i in range(idx * step, idx * step + self.n_shots):
            support_images.append(self.datasets[i]['image'])
            support_binary.append(self.datasets[i]['binary'])
            support_parts.append(self.datasets[i]['parts'])
            support_instruments.append(self.datasets[i]['instruments'])
        for j in range(idx * step + self.n_shots, (idx + 1) * step):
            query_images.append(self.datasets[j]['image'])
            query_binary.append(self.datasets[j]['binary'])
            query_parts.append(self.datasets[j]['parts'])
            query_instruments.append(self.datasets[j]['instruments'])
        sample = {
            'support_images': support_images,
            'support_binary': support_binary,
            'support_parts': support_parts,
            'support_instruments': support_instruments,
            'query_images': query_images,
            'query_binary': query_binary,
            'query_parts': query_parts,
            'query_instruments': query_instruments
        }
        return sample
