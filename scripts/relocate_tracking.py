"""Repair local MLflow artifact URIs after importing a Colab bundle."""
from pathlib import Path
import argparse
import sqlite3


def relocate(artifacts):
    artifacts = Path(artifacts).resolve()
    database = artifacts / "tracking.db"
    if not database.exists():
        return
    new = (artifacts / "mlflow_artifacts").as_uri()
    with sqlite3.connect(database) as connection:
        for table, identifier, column in (("experiments", "experiment_id", "artifact_location"), ("runs", "run_uuid", "artifact_uri")):
            for key, old in connection.execute(f"SELECT {identifier}, {column} FROM {table}"):
                if old and "mlflow_artifacts" in old:
                    suffix = old.split("mlflow_artifacts", 1)[1]
                    connection.execute(f"UPDATE {table} SET {column} = ? WHERE {identifier} = ?", (new + suffix, key))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    relocate(parser.parse_args().artifacts)
