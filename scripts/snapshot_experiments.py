"""Copy small, real experiment records into the source distribution."""
import argparse
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, default=BASE/'artifacts')
    parser.add_argument('--output', type=Path, default=BASE/'experiments/final_run')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for directory in ('configs','studies'):
        shutil.copytree(args.artifacts/directory, output/directory, dirs_exist_ok=True)
    for name in ('repair_provenance.json','detail_provenance.json','pilot_records.json','validation_comparison.json','historical_search_provenance.json','results/test/summary.json'):
        source = args.artifacts/name
        if source.exists():
            shutil.copy2(source, output/source.name)
    for name in ('optuna.db','tracking.db','historical_classifier_optuna.db'):
        source = args.artifacts/name
        if source.exists():
            with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)) as reader:
                with closing(sqlite3.connect(output/name)) as writer:
                    reader.backup(writer)
    print('Real experiment records:',output)


if __name__ == '__main__':
    main()
