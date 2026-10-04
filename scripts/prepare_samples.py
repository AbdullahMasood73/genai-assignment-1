"""Copy a few development-only clean pet examples for the UI."""
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    data = BASE / "data"
    training = json.loads((data / "manifests/pets.json").read_text())["train"]
    output = BASE / "backend/samples"
    output.mkdir(parents=True, exist_ok=True)
    choices = [training[i] for i in (0, 500, 1000, 1500)]
    for i, row in enumerate(choices):
        shutil.copy2(data / row["image"], output / f"sample-{i + 1}.png")
    (output / "attribution.json").write_text(json.dumps({"source": "https://www.robots.ox.ac.uk/~vgg/data/pets/", "license": "CC BY-SA 4.0", "split": "train", "examples": choices}, indent=2))
    print("Prepared four clean development pet samples")
