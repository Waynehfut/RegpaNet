import os
import shutil
from config import ex

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from models.fewshot import FewShotSeg
from dataloader.endovis import get_split, endo_fewshot_loader
from dataloader.transfomer import train_transform, val_transform


@ex.automain
def main(_run, _config, _log):
    # Save current code snapshot
    _log.info('### Save code snapshot ###')
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)  # save snapshots
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    _log.info(f'Code saved at {_run.observers[0].dir}/source/')

    # build few shot model
    _log.info('### Build model ###')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    model.train()
    dataname = _config['dataset']
    # load EndoVis dataset
    if dataname == 'EndoVis':
        train_file_names, val_file_names = get_split(_config['split'], _config['path']['EndoVis']['data_dir'])
        train_transfer = train_transform(_config, 1)
        problem_types = ['binary', 'parts', 'instruments']
        few_shot_loader = endo_fewshot_loader(file_names=train_file_names,
                                              shuffle=True,
                                              transform=train_transfer,
                                              problem_types=problem_types,
                                              n_ways=_config['task']['n_ways'],
                                              n_shots=_config['task']['n_shots'],
                                              n_queries=_config['task']['n_queries'],
                                              batch_size=2,
                                              num_workers=1)
    # load ROBUST dataset
    elif dataname == 'ROBUST':
        few_shot_loader = None
    else:
        raise ValueError('Cannot find the dataset definition')
    # set optimizer
    _log.info('### Set optimizer ###')
    optima = torch.optim.SGD(model.parameters(), **_config['optima'])
    scheudler = MultiStepLR(optima, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    _log.info('### Start training ###')
    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    for i_iter,sampled_batch in enumerate(few_shot_loader):
        print(i_iter)

