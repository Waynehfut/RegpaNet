import os
import shutil
from config import ex
from dataloader.endovis import endovis_simple_loader, endovis_few_shot_loader, get_split


@ex.automain
def main(_run, _config, _log):
    # Save current code snapshot
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)  # save snapshots
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    _log.info(f'Code saved at {_run.observers[0].dir}/source/')

    # load EndoVis dataset
    if _config['dataset'] == 'EndoVis':
        train_file_names, val_file_names = get_split(_config['split'], _config['path']['EndoVis']['data_dir'])
        _log.info(f'load {_config["dataset"]} '
                  f'with {len(train_file_names)} train sample '
                  f'and {len(val_file_names)} val sample')
        few_shot_loader = endovis_few_shot_loader(train_file_names, val_file_names,
                                                  _config['problem_type'],
                                                  _config['task']['n_ways'],
                                                  _config['task']['n_shots'],
                                                  _config['task']['n_queries'],
                                                  _config['batch_size'])
