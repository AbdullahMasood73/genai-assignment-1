"""Separate validation-led improvement; do not evaluate test data automatically."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=BASE, check=True)


def backup(root, task):
    output = BASE / 'delivery_detail' / f'detail_after_{task}.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and 'trials' not in path.relative_to(root).parts:
                archive.write(path, path.relative_to(root).as_posix())
    print('CHECKPOINT_BACKUP_READY', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=BASE/'artifacts_detail_pilot')
    parser.add_argument('--pilot-record', type=Path,
                        help='Saved real pilot history/configuration if the free runtime was deleted')
    parser.add_argument('--artifacts', type=Path, default=BASE/'artifacts_detail')
    parser.add_argument('--cpu-budget', action='store_true',
                        help='Fixed deadline-aware CPU schedule after free GPU quota')
    parser.add_argument('--cpu-channels-last',action='store_true',
                        help='Use the locally benchmarked tensor layout for CPU execution')
    parser.add_argument('--historical-optuna', type=Path,
                        help='Original database containing the retained classifier study')
    args = parser.parse_args()
    sys.path.insert(0, str(BASE))
    from restoration.train import load_checkpoint
    if args.pilot_record:
        evidence = json.loads(args.pilot_record.read_text())
        if evidence.get('split') != 'validation' or not evidence.get('full_split'):
            raise ValueError('A complete validation pilot record is required')
        pilot = {'best': min(row['validation_objective'] for row in evidence['history']),
                 'config': evidence['config']}
    else:
        pilot = load_checkpoint(args.pilot/'checkpoints/universal/best.pt')
    previous = load_checkpoint(args.baseline/'checkpoints/universal/best.pt')
    if not pilot['best'] < previous['best']:
        raise RuntimeError('Detail pilot does not beat the previous validation objective')
    root = args.artifacts.resolve()
    if root == args.baseline.resolve() or root == (BASE/'artifacts').resolve():
        raise ValueError('Use a separate improvement directory')
    for task in ('classifier',):
        destination = root/'checkpoints'/task
        if not destination.exists():
            shutil.copytree(args.baseline/'checkpoints'/task, destination)
        for folder in ('configs', 'studies'):
            target = root/folder/f'{task}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(args.baseline/folder/f'{task}.json', target)
    if not (root/'tracking.db').exists():
        shutil.copy2(args.baseline/'tracking.db', root/'tracking.db')
        shutil.copytree(args.baseline/'mlflow_artifacts', root/'mlflow_artifacts', dirs_exist_ok=True)
        from scripts.relocate_tracking import relocate
        relocate(root)
    if args.historical_optuna and not (root/'historical_classifier_optuna.db').exists():
        import sqlite3
        with sqlite3.connect(args.historical_optuna.resolve().as_uri()+'?mode=ro',uri=True) as reader:
            if not reader.execute("SELECT 1 FROM studies WHERE study_name='classifier'").fetchone():
                raise ValueError('Historical database does not contain the classifier study')
            with sqlite3.connect(root/'historical_classifier_optuna.db') as writer:
                reader.backup(writer)
    provenance = {'status':'training','previous_test_previously_observed':True,
        'selection':'validation only; no automatic test evaluation',
        'pilot_best':pilot['best'],'previous_universal_best':previous['best'],
        'pilot_config':pilot['config'],
        'pilot_weights_available':not bool(args.pilot_record),
        'runtime_recovery':'Fresh CPU training from saved validation evidence after free Colab GPU quota; pilot weights were lost' if args.pilot_record else 'Free Colab GPU',
        'architecture':'16x16 compressed latent, local residual refinement, no cross-encoder-decoder skips',
        'upsampling_initialization':'ICNR phase matched in fresh final runs; selected pilot used standard initialization',
        'classifier':'retained from the previous validation-selected run',
        'gan':'new normalized U-Net/PatchGAN with learned upsampling; separate Optuna study and full training'}
    pilot_records = []
    if args.pilot_record:
        pilot_records = evidence.get('all_pilots', [{'pilot':'selected_saved_pilot',
            'config':evidence['config'], 'history':evidence['history'],
            'best':min(evidence['history'],key=lambda row:row['validation_objective']),
            'selected':True}])
    for folder in sorted(BASE.glob('artifacts_detail_pilot*')):
        history = folder/'checkpoints/universal/history.json'
        if history.exists():
            rows = json.loads(history.read_text())
            pilot_records.append({'pilot':folder.name,
                'config':json.loads((folder/'pilot_config.json').read_text()),
                'best':min(rows,key=lambda row:row['validation_objective']),
                'history':rows,'selected':folder.resolve()==args.pilot.resolve()})
    (root/'pilot_records.json').write_text(json.dumps(pilot_records,indent=2))
    import optuna
    for task in ('universal', 'specialists'):
        study = optuna.create_study(study_name=task,
            storage='sqlite:///'+(root/'optuna.db').as_posix(),
            direction='minimize', load_if_exists=True)
        if not study.trials:
            study.enqueue_trial({name:pilot['config'][name] for name in
                                 ('batch_size','lr','base','dropout','latent','alpha')})
    gan_study = optuna.create_study(study_name='gan',
        storage='sqlite:///'+(root/'optuna.db').as_posix(),
        direction='minimize', load_if_exists=True)
    if not gan_study.trials:
        gan_study.enqueue_trial({'batch_size':32,'lr':0.0002,'base':24,'dropout':0.05,
            'discriminator_lr':0.0002,'embedding':16,'l1_weight':100.0})
    tasks = [('universal',24,True),('specialists',20,True),('soft',12,True),('gan',60,True)] if args.cpu_budget else [('universal',50,True),('specialists',40,True),('soft',25,True),('gan',100,True)]
    provenance['fixed_schedule'] = {task:epochs for task,epochs,_ in tasks}
    trial_epochs, train_batches, validation_batches = (3,40,40) if args.cpu_budget else (6,60,80)
    provenance['trial_budget'] = {'epochs':trial_epochs,'train_batches':train_batches,'validation_batches':validation_batches}
    provenance['cpu_execution_layout'] = 'channels_last' if args.cpu_channels_last else 'contiguous'
    status = root/'detail_provenance.json'
    for task, epochs, tune in tasks:
        if task == 'gan':
            run('-m','restoration.prepare','--dataset','fs2k')
        status.write_text(json.dumps(provenance,indent=2))
        options = ['-m','restoration.train','--task',task,'--artifacts',root,
            '--mode','both' if tune else 'train','--epochs',epochs,'--workers','0' if args.cpu_budget else '2',
            '--trials','3','--trial-epochs',trial_epochs,'--trial-train-batches',train_batches,
            '--trial-validation-batches',validation_batches]
        if task in ('universal','specialists'):
            options += ['--detail']
            if pilot['config'].get('detail_bn'):
                options += ['--detail-bn']
            if pilot['config'].get('detail_shuffle'):
                options += ['--detail-shuffle']
        if task == 'gan':
            options += ['--gan-refined']
        if args.cpu_channels_last:
            options += ['--cpu-channels-last']
        run(*options)
        provenance['last_completed_task'] = task
        status.write_text(json.dumps(provenance,indent=2))
        backup(root, task)
    run('-m','restoration.evaluate','--artifacts',root,'--split','validation',
        *(['--cpu-channels-last'] if args.cpu_channels_last else []))
    run('scripts/compare_validation.py',
        '--baseline',args.baseline/'results/validation/summary.json',
        '--candidate',root/'results/validation/summary.json',
        '--output',root/'validation_comparison.json')
    run('-m','restoration.export','--artifacts',root)
    provenance['status'] = 'validation_evaluated_and_exported'
    status.write_text(json.dumps(provenance,indent=2))
    backup(root, 'validation')


if __name__ == '__main__':
    main()
