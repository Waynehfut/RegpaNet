import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"
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
    _log.info(('Let us using {} GPUs'.format(torch.cuda.device_count())))
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
    # model = nn.DataParallel(model.cuda())
    model = model.cuda()
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
                                              batch_size=_config['batch_size'],
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
    for i_iter, sampled_batch in enumerate(few_shot_loader):
        support_imgs = [[img.cuda() for img in way] for way in sampled_batch['support_images']]
        support_binary_masks = [[img.float().cuda() for img in way] for way in sampled_batch['support_binary']]
        support_parts_masks = [[img.float().cuda() for img in way] for way in sampled_batch['support_parts']]
        support_instruments_masks = [[img.float().cuda() for img in way] for way in
                                     sampled_batch['support_instruments']]
        query_imgs = [[img.cuda() for img in way] for way in sampled_batch['query_images']]
        query_binary_masks = torch.cat([way.float().cuda() for way in sampled_batch['query_binary']],
                                       dim=0)
        query_parts_masks = torch.cat([way.float().cuda() for way in sampled_batch['query_parts']],
                                       dim=0)
        query_instruments_masks = torch.cat([way.float().cuda() for way in sampled_batch['query_instruments']],
                                       dim=0)
        optima.zero_grad()
        query_pred, align_loss = model(support_imgs, support_binary_masks, query_imgs)
        query_loss = criterion(query_pred, query_binary_masks)
        loss = query_loss + align_loss * _config['align_loss_scalar']
        loss.backward()
        optima.step()
        scheudler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss
        _log.info('loss', query_loss, 'align_loss', align_loss)

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter + 1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

        _log.info('###### Saving final model ######')
        torch.save(model.state_dict(),
                   os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
