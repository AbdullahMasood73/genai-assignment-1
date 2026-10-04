"""Portable source/training/result packages; excludes environments and private files."""
import argparse
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]
EXCLUDED = {"node_modules", ".venv", "__pycache__", ".pytest_cache", "dist", "generated", ".git"}


def source_files():
    for name in ("restoration", "backend", "frontend", "tests", "scripts", "docs", "report", "experiments"):
        for path in (BASE / name).rglob("*"):
            relative = path.relative_to(BASE)
            if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
                continue
            if "screenshots" in relative.parts or path.suffix in (".log", ".aux", ".out", ".pdf", ".tsbuildinfo"):
                continue
            yield path
    for name in ("README.md", ".gitignore", ".dockerignore", "compose.yaml", "pyproject.toml", "requirements.txt", "requirements-train.txt", "requirements-lock.txt"):
        path = BASE / name
        if path.exists():
            yield path


def zip_files(target, paths, root):
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(root).as_posix())


def package_models(artifacts, target):
    artifacts = Path(artifacts)
    manifest = artifacts / "models/manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Export verified real checkpoints before packaging a model bundle")
    paths = [p for p in artifacts.rglob("*") if p.is_file() and "trials" not in p.relative_to(artifacts).parts and p.suffix != ".tmp"]
    zip_files(Path(target), paths, artifacts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=BASE / "artifacts")
    parser.add_argument("--output", type=Path, default=BASE / "delivery")
    args = parser.parse_args()
    source = list(source_files())
    source += [p for folder in ('report/generated', 'docs/screenshots')
               for p in (BASE / folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    zip_files(args.output / "source.zip", source, BASE)
    if args.data:
        data = BASE / "data"
        paths = [p for name in ("pets", "fs2k", "manifests") for p in (data / name).rglob("*") if p.is_file()]
        if (data / "preparation.json").exists():
            paths.append(data / "preparation.json")
        zip_files(args.output / "prepared_data.zip", paths, data)
    if args.models:
        package_models(args.artifacts, args.output / "trained_bundle.zip")
    print(args.output)


if __name__ == "__main__":
    main()
