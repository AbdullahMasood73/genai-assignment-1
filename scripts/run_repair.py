"""Separate validation-led repair; preserve the original failed experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parents[1]


def run(*arguments):
    command = [sys.executable, *map(str, arguments)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=BASE, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, default=BASE/'artifacts_repair')
    parser.add_argument('--pilot', type=Path, default=BASE/'artifacts_repair_pilot_spatial')
    parser.add_argument('--selected-settings', type=Path, required=True,
                        help='Original classifier/GAN validation-selected configs and study records')
    args = parser.parse_args()
    if args.artifacts.resolve() == (BASE/'artifacts').resolve():
        raise ValueError('Repair must preserve the original artifacts directory')
    sys.path.insert(0, str(BASE))
    from restoration.train import load_checkpoint
    pilot = load_checkpoint(args.pilot/'checkpoints/universal/best.pt')
    score = float(pilot['best'])
    if not score < 0.30:
        raise RuntimeError(f'Pilot validation objective {score:.6f} does not justify the full repair run')
    settings = json.loads(args.selected_settings.read_text())
    root = args.artifacts.resolve()
    (root/'configs').mkdir(parents=True, exist_ok=True)
    (root/'studies').mkdir(exist_ok=True)
    provenance = {
        'reason': 'Near-constant outputs in the first-run validation examples',
        'pilot_validation_objective': score,
        'pilot_history': json.loads((args.pilot/'checkpoints/universal/history.json').read_text()),
        'change': 'Compressed spatial latent; GroupNorm/LeakyReLU; bilinear resize-convolution decoder',
        'original_test_previously_observed': True,
        'selection': 'Repair decisions and checkpoint selection use validation only',
        'classifier_gan': 'Retrained with original validation-selected settings; search records retained',
        'status': 'training',
    }
    status = root/'repair_provenance.json'
    status.write_text(json.dumps(provenance, indent=2))
    import optuna
    for task in ('universal', 'specialists'):
        study = optuna.create_study(study_name=task,
            storage='sqlite:///' + (root/'optuna.db').as_posix(),
            load_if_exists=True, direction='minimize')
        if not study.trials:
            study.enqueue_trial({name: pilot['config'][name] for name in
                                 ('batch_size','lr','base','dropout','latent','alpha')})
    for task in ('classifier', 'gan'):
        (root/'configs'/f'{task}.json').write_text(json.dumps(settings[task]['config'], indent=2))
        (root/'studies'/f'{task}.json').write_text(json.dumps(settings[task]['study'], indent=2))
    for task, epochs, tune in [('universal',20,True), ('classifier',12,False),
                               ('specialists',20,True), ('soft',15,True), ('gan',50,False)]:
        options = ['-m','restoration.train','--task',task,'--artifacts',root,
                   '--mode','both' if tune else 'train','--epochs',epochs,
                   '--workers','2','--trials','4','--trial-epochs','3']
        if task in ('universal','specialists'):
            options += ['--normalized','--spatial']
        run(*options)
        provenance['last_completed_task'] = task
        status.write_text(json.dumps(provenance, indent=2))
    run('-m','restoration.evaluate','--artifacts',root,'--split','validation')
    run('-m','restoration.export','--artifacts',root)
    # Fixed schedule completed; evaluate these frozen validation-selected models once.
    run('-m','restoration.evaluate','--artifacts',root,'--split','test')
    provenance['status'] = 'evaluated_and_exported'
    status.write_text(json.dumps(provenance, indent=2))
    output = BASE/'delivery_repair'
    output.mkdir(exist_ok=True)
    for label, prefixes in [('models',('models/',)),
                            ('results',('results/','configs/','studies/','mlflow_artifacts/',
                                        'tracking.db','optuna.db','repair_provenance.json'))]:
        with zipfile.ZipFile(output/f'repair_{label}.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.relative_to(root).as_posix().startswith(prefixes):
                    archive.write(path,path.relative_to(root).as_posix())
    # Preserve small inference checkpoints too; optimizer/RNG states are excluded.
    import torch
    weights = output/'inference_checkpoints'
    for path in sorted((root/'checkpoints').glob('*/best.pt')):
        saved = load_checkpoint(path)
        compact = {k:v for k,v in saved.items() if k not in
                   ('optimizer','scaler','discriminator','d_optimizer',
                    'python_rng','numpy_rng','torch_rng','cuda_rng')}
        destination = weights/path.relative_to(root/'checkpoints')
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(compact,destination)
    with zipfile.ZipFile(output/'repair_inference_checkpoints.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(weights.rglob('*.pt')):
            archive.write(path,'checkpoints/'+path.relative_to(weights).as_posix())
    from scripts.package_project import package_models
    package_models(root, output/'repair_full_bundle.zip')
    print('Repair outputs ready:',output,flush=True)


if __name__ == '__main__':
    main()
