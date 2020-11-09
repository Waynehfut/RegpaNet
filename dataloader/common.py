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
    def __init__(self, dataset, n_ways, n_shots, n_queries):
        """
        Transfer simple dataset into N_ways N_shots datasets, with N_queries.

        :param dataset: Simple dataset
        :param n_ways: Tasks, normally will be 1, for EndoVis can be 1,2,3
        :param n_shots: Train sample in a way
        :param n_queries: Test sample in a way
        """
        self.dataset = dataset
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        if len(self.dataset[0]) < (self.n_ways + 1):
            raise ValueError(f'Dataset not support the way larger than {self.n_ways} ways')

    def __len__(self):
        int(len(self.dataset) / (self.n_shots + self.n_queries))

    def __getitem__(self, idx):
        step = self.n_shots + self.n_queries
        support_set = []
        query_set = []
        for i in range(idx * step, idx * step + self.n_shots):
            support_set.append(self.dataset[i])
        for j in range(idx * step + self.n_shots, (idx + 1) * step):
            query_set.append(self.dataset[j])
        sample = {'support': support_set,
                  'query': query_set}
        return sample
