import os
import shutil
from config import ex
from dataloader.endovis import EndoVisDataSet, make_simple_loader, make_few_shot_loader, get_split


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
        train_loader = make_simple_loader(train_file_names, shuffle=True, transform=None,
                                          problem_type=_config['problem_type'],
                                          batch_size=_config['batch_size'])
        val_loader = make_simple_loader(val_file_names, transform=None, problem_type=_config['problem_type'],
                                        batch_size=_config['batch_size'])
        _log.info(f'load {_config["dataset"]} with {len(train_loader)} train sample and {len(val_loader)} val sample')
