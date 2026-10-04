"""Download official data, resize once, create immutable split/corruption manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps

from .corruptions import CLASSES, LEVELS, config

PET_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/"
FS2K_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_extract(archive: Path, destination: Path):
    """Reject traversal and symlinks in downloaded archives."""
    destination = destination.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as reader:
            for member in reader.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            reader.extractall(destination)
    else:
        with tarfile.open(archive) as reader:
            reader.extractall(destination, filter="data")


def resize(source, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BICUBIC).save(destination)


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(target)


def prepare_pets(root: Path, download_data=True):
    raw = root / "raw" / "pets"
    raw.mkdir(parents=True, exist_ok=True)
    for filename in ("images.tar.gz", "annotations.tar.gz"):
        if not (raw / filename.removesuffix(".tar.gz")).exists():
            if download_data:
                download(PET_URL + filename, raw / filename)
            if not (raw / filename).exists():
                raise FileNotFoundError(f"Supply {raw / filename}")
            safe_extract(raw / filename, raw)
    names = {}
    for split, filename in (("development", "trainval.txt"), ("test", "test.txt")):
        names[split] = [line.split()[0] for line in (raw / "annotations" / filename).read_text().splitlines()
                        if line.strip() and not line.startswith("#")]
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(names["development"]))
    # sklearn-compatible 20% rounding; seed and actual names are persisted.
    validation_count = int(np.ceil(0.2 * len(indices)))
    split_names = {"validation": [names["development"][i] for i in indices[:validation_count]],
                   "train": [names["development"][i] for i in indices[validation_count:]],
                   "test": names["test"]}
    if set(split_names["train"]) & set(split_names["test"]):
        raise ValueError("Official train/test overlap")
    for group in split_names.values():
        for name in group:
            resize(raw / "images" / f"{name}.jpg", root / "pets" / f"{name}.png")
    manifest = {split: [{"id": name, "image": f"pets/{name}.png"} for name in group]
                for split, group in split_names.items()}
    write_json(root / "manifests" / "pets.json", manifest)
    for split in ("validation", "test"):
        rows = []
        for i, row in enumerate(manifest[split]):
            for kind in CLASSES:
                for level in (["clean"] if kind == "clean" else LEVELS):
                    seed = 42 + i * 31 + CLASSES.index(kind) * 5 + (LEVELS.index(level) if level in LEVELS else 0)
                    cfg = config(kind, seed, None if kind == "clean" else level)
                    rows.append({**row, "corruption": cfg})
        write_json(root / "manifests" / f"pets_{split}.json", rows)
    return {split: len(group) for split, group in manifest.items()}


def find_image(stem):
    if stem.is_file():
        return stem
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        path = stem.with_suffix(suffix)
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def prepare_fs2k(root: Path, source: Path | None, download_data=True):
    if source is None:
        raw = root / "raw"
        archive = raw / "FS2K.zip"
        raw.mkdir(parents=True, exist_ok=True)
        annotations = list(raw.rglob("anno_train.json"))
        if not annotations:
            if download_data and not archive.exists():
                import gdown
                if not gdown.download(id=FS2K_ID, output=str(archive), quiet=False):
                    raise RuntimeError("Official FS2K download unavailable; supply --fs2k-root")
            if not archive.exists():
                raise FileNotFoundError("FS2K.zip missing; supply --fs2k-root")
            safe_extract(archive, raw)
            annotations = list(raw.rglob("anno_train.json"))
        if len(annotations) != 1:
            raise ValueError("Expected exactly one FS2K annotation root")
        source = annotations[0].parent
    official_train = json.loads((source / "anno_train.json").read_text())
    official_test = json.loads((source / "anno_test.json").read_text())
    if len(official_train) != 1058 or len(official_test) != 1046:
        raise ValueError("Expected official FS2K splits: 1058 training, 1046 test pairs")
    from sklearn.model_selection import train_test_split
    train, validation = train_test_split(official_train, test_size=0.15, random_state=42,
                                         stratify=[row["style"] for row in official_train])
    if {row["image_name"] for row in official_train} & {row["image_name"] for row in official_test}:
        raise ValueError("FS2K split overlap")
    manifest = {}
    for split, group in (("train", train), ("validation", validation), ("test", official_test)):
        manifest[split] = []
        for row in group:
            name = row["image_name"]
            style = int(row["style"])
            if style not in (0, 1, 2):
                raise ValueError(f"Unknown FS2K style: {style}")
            photo = find_image(source / "photo" / name)
            sketch_name = name.replace("photo", "sketch").replace("image", "sketch")
            sketch = find_image(source / "sketch" / sketch_name)
            identifier = name.replace("/", "_")
            photo_dest = f"fs2k/photo/{identifier}.png"
            sketch_dest = f"fs2k/sketch/{identifier}.png"
            resize(photo, root / photo_dest)
            resize(sketch, root / sketch_dest)
            manifest[split].append({"id": identifier, "image": photo_dest,
                                    "target": sketch_dest, "style": style})
    write_json(root / "manifests" / "fs2k.json", manifest)
    return {split: {"pairs": len(group), "styles": [sum(row["style"] == s for row in group) for s in range(3)]}
            for split, group in manifest.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["pets", "fs2k", "all"], default="all")
    parser.add_argument("--fs2k-root", type=Path)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    summary = {"seed": 42, "image_size": 128}
    if args.dataset in ("pets", "all"):
        summary["pets"] = prepare_pets(args.data, not args.no_download)
    if args.dataset in ("fs2k", "all"):
        summary["fs2k"] = prepare_fs2k(args.data, args.fs2k_root, not args.no_download)
    summary["manifest_sha256"] = {p.name: checksum(p) for p in sorted((args.data / "manifests").glob("*.json"))}
    write_json(args.data / "preparation.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
