"""Consistent database snapshots and checkpoint backups for interrupted Colab runs."""
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing


def snapshot(source, destination, task=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Backup must be outside the local artifacts directory")
    destination.mkdir(parents=True, exist_ok=True)
    directories = ["configs", "studies", "mlflow_artifacts", "models", "results"]
    directories.append(f"checkpoints/{task}" if task else "checkpoints")
    for name in directories:
        folder = source / name
        if folder.exists():
            shutil.copytree(folder, destination / name, dirs_exist_ok=True)
    for name in ("tracking.db", "optuna.db"):
        database = source / name
        if database.exists():
            target = destination / (name + ".tmp")
            with closing(sqlite3.connect(database)) as original, closing(sqlite3.connect(target)) as copy:
                original.backup(copy)
            target.replace(destination / name)
