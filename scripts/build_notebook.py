"""Generate a Colab notebook with embedded source; no Git clone or second upload."""
import base64
from io import BytesIO
import json
from pathlib import Path
import textwrap
import zipfile

from package_project import BASE, source_files


def cell(kind, source, identifier):
    item = {"cell_type": kind, "metadata": {"id": identifier}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def main():
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_files()):
            archive.write(path, path.relative_to(BASE).as_posix())
    source = base64.b64encode(stream.getvalue()).decode()
    cells = [cell("markdown", """
        # Generative AI Assignment 1 — resumable free-GPU training

        This notebook contains the complete prepared source. It does not clone a Git
        repository or require any paid service. Select **Runtime → Change runtime type →
        T4 GPU** (or another available free GPU), then run the preparation and training cells in order. Google Drive access
        is not required. Download checkpoints before the temporary runtime is deleted.

        The notebook downloads the two official datasets, creates deterministic splits,
        runs five Optuna studies, trains seven inference models, evaluates frozen
        checkpoints, checks ONNX parity and saves an export bundle. Model quality is not
        guaranteed by short training. Review real validation evidence before claiming
        completion. Reconnect and Run all to resume while runtime files still exist;
        after runtime deletion, restore a downloaded checkpoint bundle first.

        No keep-alive tricks or quota bypasses are used. Do not run model-serving apps
        in this free Colab runtime; the final application runs locally in Docker.
        """, "intro"), cell("code", f"""
        import base64, io, os, pathlib, zipfile, subprocess, sys, json, shutil
        PROJECT = pathlib.Path('/content/restoration-lab')
        PROJECT.mkdir(exist_ok=True)
        SOURCE = '{source}'
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as archive:
            archive.extractall(PROJECT)
        os.chdir(PROJECT)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--timeout', '120', '-r', 'requirements-train.txt'], check=True)
        subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(), 'Select a free GPU runtime before training'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__)"], check=True)
        print('Source and dependencies ready.')
        """, "bootstrap"), cell("code", """
        # Temporary runtime storage only. No Google Drive access.
        PERSISTENT = pathlib.Path('/content/assignment-downloads')
        PERSISTENT.mkdir(parents=True, exist_ok=True)
        ARTIFACTS = PROJECT / 'artifacts'
        BACKUP = PERSISTENT / 'artifacts'
        if BACKUP.exists():
            shutil.copytree(BACKUP, ARTIFACTS, dirs_exist_ok=True)
            subprocess.run([sys.executable, 'scripts/relocate_tracking.py'], check=True)
        ARTIFACTS.mkdir(exist_ok=True)
        # Fixed reproduction budget. Change it only before final model selection,
        # record the change, and never choose it from official test results.
        TRIALS = 3
        TRIAL_EPOCHS = 3
        EPOCHS = {'universal': 24, 'classifier': 12, 'specialists': 20, 'soft': 12, 'gan': 60}
        print('Each final epoch copies checkpoints and databases to', BACKUP)
        print('Download results before this temporary runtime is deleted.')
        """, "persistence"), cell("code", """
        # Resume from a single prepared-data archive, or download verified official sources.
        cache = PERSISTENT / 'prepared_data.zip'
        if cache.exists() and not (PROJECT / 'data/manifests/fs2k.json').exists():
            from restoration.prepare import safe_extract
            safe_extract(cache, PROJECT / 'data')
        if not (PROJECT / 'data/manifests/fs2k.json').exists():
            subprocess.run([sys.executable, '-m', 'restoration.prepare', '--dataset', 'all'], check=True)
        subprocess.run([sys.executable, 'scripts/verify_data.py'], check=True)
        if not cache.exists():
            subprocess.run([sys.executable, 'scripts/package_project.py', '--data'], check=True)
            shutil.copy2(PROJECT / 'delivery/prepared_data.zip', cache)
        print((PROJECT / 'data/preparation.json').read_text())
        """, "data")]
    cells.append(cell("code", """
        # A validation-tested configuration is investigated alongside TPE trials.
        # This seeds the study; Optuna still selects the final configuration.
        import optuna
        seeds = {
            'universal': {'batch_size':32,'lr':.0003,'base':24,'dropout':0.,'latent':8192,'alpha':.8},
            'specialists': {'batch_size':32,'lr':.0003,'base':24,'dropout':0.,'latent':8192,'alpha':.8},
            'gan': {'batch_size':32,'lr':.0002,'base':24,'dropout':.05,
                    'discriminator_lr':.0002,'embedding':16,'l1_weight':100.}
        }
        for task, seed_config in seeds.items():
            study = optuna.create_study(study_name=task,
                storage='sqlite:///' + (ARTIFACTS/'optuna.db').as_posix(),
                direction='minimize', load_if_exists=True)
            if not study.trials:
                study.enqueue_trial(seed_config)
        """, "seed_studies"))
    for task in ("universal", "classifier", "specialists", "soft", "gan"):
        cells.append(cell("markdown", f"## Train {task}\nSelected settings are retrained from scratch. Reruns resume saved final epochs; study trial counts are cumulative.", "heading_" + task))
        architecture_options = "'--detail', '--detail-bn', '--detail-shuffle'," if task in ('universal','specialists') else ("'--gan-refined'," if task == 'gan' else '')
        cells.append(cell("code", f"""
            subprocess.run([sys.executable, '-m', 'restoration.train', '--task', '{task}',
                            {architecture_options}
                            '--mode', 'both', '--trials', str(TRIALS), '--trial-epochs', str(TRIAL_EPOCHS),
                            '--trial-train-batches', '40', '--trial-validation-batches', '40',
                            '--epochs', str(EPOCHS['{task}']), '--backup', str(BACKUP)], check=True)
            print('Completed {task}.')
            """, "train_" + task))
    cells += [cell("markdown", """
        ## Validation evidence and export
        The remaining cells evaluate validation data and export the selected checkpoints.
        Final test evaluation is deliberately disabled until models are frozen. Earlier
        assignment attempts already observed the official test; disclose any re-evaluation. Do not repeatedly use test
        results to tune configurations. If you change the training schedule, disclose
        that decision and avoid treating the official test as a validation set.
        """, "evaluation_heading"), cell("code", """
        subprocess.run([sys.executable, '-m', 'restoration.evaluate', '--split', 'validation'], check=True)
        from IPython.display import display, Image
        display(Image(filename=str(ARTIFACTS / 'results/validation/confusion.png')))
        display(Image(filename=str(ARTIFACTS / 'results/validation/face_representative.png')))
        """, "validation"), cell("code", """
        # Set True only after reviewing validation and freezing the selected models.
        RUN_FROZEN_TEST = False
        if RUN_FROZEN_TEST:
            subprocess.run([sys.executable, '-m', 'restoration.evaluate', '--split', 'test'], check=True)
        subprocess.run([sys.executable, '-m', 'restoration.export'], check=True)
        subprocess.run([sys.executable, 'scripts/build_report.py', '--draft'], check=True)
        from restoration.backup import snapshot
        snapshot(ARTIFACTS, BACKUP)
        subprocess.run([sys.executable, 'scripts/package_project.py', '--models'], check=True)
        shutil.copy2(PROJECT / 'delivery/trained_bundle.zip', PERSISTENT / 'trained_bundle.zip')
        print('Final model/results bundle saved:', PERSISTENT / 'trained_bundle.zip')
        print('Report source now includes the real results, but Stitch/video/container evidence still needs completion.')
        """, "final"), cell("code", """
        # Download the bundle before this temporary runtime is deleted.
        from google.colab import files
        files.download(str(PERSISTENT / 'trained_bundle.zip'))
        """, "download")]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "colab": {"name": "Assignment1_Train.ipynb", "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
    target = BASE / "notebooks/Assignment1_Train.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"{target} ({target.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
