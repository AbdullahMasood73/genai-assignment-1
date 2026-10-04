"""Check official memberships, immutable manifests and pairing without model evaluation."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from restoration.prepare import checksum
from PIL import Image


def verify(root):
    root = Path(root).resolve()
    preparation = json.loads((root / "preparation.json").read_text())
    for name, expected in preparation["manifest_sha256"].items():
        assert checksum(root / "manifests" / name) == expected, f"Manifest changed: {name}"
    pets = json.loads((root / "manifests/pets.json").read_text())
    fs2k = json.loads((root / "manifests/fs2k.json").read_text())
    assert [len(pets[n]) for n in ("train", "validation", "test")] == [2944, 736, 3669]
    assert [len(fs2k[n]) for n in ("train", "validation", "test")] == [899, 159, 1046]
    for groups in (pets, fs2k):
        members = [{r["id"] for r in groups[n]} for n in ("train", "validation", "test")]
        assert not any(members[i] & members[j] for i in range(3) for j in range(i + 1, 3))
        for rows in groups.values():
            for row in rows:
                for field in ("image", "target"):
                    if field in row:
                        path = (root / row[field]).resolve()
                        assert path.is_relative_to(root) and path.is_file(), f"Missing/unsafe image: {path}"
    for split in ("validation", "test"):
        rows = json.loads((root / f"manifests/pets_{split}.json").read_text())
        assert len(rows) == 10 * len(pets[split])
        conditions = {}
        for row in rows:
            cfg = row["corruption"]
            conditions.setdefault(row["id"], set()).add((cfg["kind"], cfg["level"]))
            if cfg["kind"] == "occlusion":
                fraction = sum(w * h for _, _, w, h in cfg["rectangles"]) / 128**2
                assert abs(fraction - cfg["requested_area"]) < .005
        assert all(len(v) == 10 for v in conditions.values())
        assert set(conditions) == {r["id"] for r in pets[split]}
    # Inspect only development examples for image dimensions; test memberships are metadata checks.
    for groups in (pets, fs2k):
        for row in groups["train"][:8]:
            with Image.open(root / row["image"]) as image:
                assert image.mode == "RGB" and image.size == (128, 128)
    print("Official split membership, all image paths, corruption manifests and development image shapes verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=BASE / "data")
    verify(parser.parse_args().data)
