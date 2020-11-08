# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       config.py
   Description :     Init config by the sacred
   Author :          HAO
   Create by :       PyCharm
   Check status:     https://waynehfut.com
-------------------------------------------------
"""

import os
import re
import glob
import itertools
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
# sacred.SETTINGS.CAPTURE_MODE = 'no' # control the info output
ex = Experiment('RegpaNet')
ex.captured_out_filter = apply_backspaces_and_linefeeds
source_folders = ['.', './dataloader', './models', './utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))  # snap all python file
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    mode = 'train'  # or 'test'
    problem_type = 'binary'  # or 'parts', 'instruments'
    split = 0  # or 1,2,3
    if mode == 'train':
        dataset = 'EndoVis'  # or 'ROBUST'
        n_steps = 30000
        label_sets = 0
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        print_interval = 100
        save_pred_every = 10000
        model = {
            'align': True
        }
        task = {
            'n_ways': 1,  # For EndoVis can be 1,2,3 with corresponding task (binary, parts and instruments)
            'n_shots': 1,
            'n_queries': 1,
        }
        optima = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
    elif mode == 'test':
        is_train = False
        snapshot = './run/RegpaNet_EndoVis_align_sets_0_1way_1shot_[train]/1/snapshots/30000.pth'
        if 'EndoVis' in snapshot:
            dataset = 'EndoVis'
        elif 'ROBUST' in snapshot:
            dataset = 'ROBUST'
        else:
            raise ValueError("Wrong snapshot name")
        model = {}
        for key in ['align', ]:
            model[key] = key in snapshot

        label_sets = int(snapshot.split('_sets_')[1][0])

        task = {
            'n_ways': int(re.search("[0-9]+way", snapshot).group(0)[:-3]),
            'n_shots': int(re.search("[0-9]+shot", snapshot).group(0)[:-4]),
            'n_queries': 1,
        }
    else:
        raise ValueError('Wrong configure for mode')
    exp_str = '_'.join(
        [dataset, ]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}_{problem_type}]']
    )
    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/xxx.pth',
        'EndoVis': {'data_dir': './data/EndoVis',
                    'data_split': 'train'},
        # 'ROBUST': {}
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
