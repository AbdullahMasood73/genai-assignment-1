"""Final evaluation from immutable manifests; writes real tables and evidence.

Test mode is explicitly requested and records checkpoint hashes. It is never
called by hyperparameter selection. Missing checkpoints cause a hard failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'tmp/matplotlib'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 13, 'axes.labelsize': 13,
                          'axes.titlesize': 14, 'xtick.labelsize': 12,
                          'ytick.labelsize': 12})
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix
import torch

from .corruptions import CLASSES, read_rgb
from .data import Pets, Faces
from .models import ssim
from .prepare import checksum, write_json
from .train import load_checkpoint, model_from_checkpoint, psnr


def metric_values(output, target):
    return {"l1": (output - target).abs().mean((1, 2, 3)).cpu().tolist(),
            "psnr": psnr(output, target).cpu().tolist(), "ssim": ssim(output, target).cpu().tolist()}


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_grid(path, examples, title):
    width, height = 512, 164
    canvas = Image.new("RGB", (width, 52 + height * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 16)
    title_font = ImageFont.truetype(str(font_path), 20)
    draw.text((6, 4), title, fill="black", font=title_font)
    for i, label in enumerate(("Clean target", "Input", "Output", "Absolute error")):
        draw.text((128 * i + 4, 30), label, fill="black", font=font)
    for i, (description, values) in enumerate(examples):
        y = 52 + height * i
        while draw.textlength(description, font=font) > width - 12:
            description = description[:-1]
        draw.text((6, y), description, fill="black", font=font)
        for column, value in enumerate(values):
            array = (value.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (column * 128, y + 24))
    canvas.save(path)


def representative_rows(rows):
    """Fixed seed/condition coverage, selected without inspecting output quality."""
    conditions = [('clean','clean')]+[(kind,level) for kind in CLASSES[1:]
                                     for level in ('low','medium','high')]
    conditions += [('salt','high'),('occlusion','high')]
    rng = np.random.default_rng(42)
    selected, seen = [], set()
    for kind,level in conditions:
        group = [row for row in rows if row['corruption']==kind and row['severity']==level]
        fresh = [row for row in group if row['id'] not in seen]
        group = fresh or group
        if group:
            row = group[int(rng.integers(len(group)))]
            selected.append(row)
            seen.add(row['id'])
    return selected


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    memory_format = torch.channels_last if device.type == 'cpu' and args.cpu_channels_last else torch.contiguous_format
    names = ("universal", "classifier", "salt", "blur", "occlusion", "soft", "gan")
    models, frozen = {}, {}
    for name in names:
        path = args.artifacts / "checkpoints" / name / "best.pt"
        saved = load_checkpoint(path)
        if saved.get("synthetic"):
            raise ValueError("Synthetic checkpoints cannot be used as assignment results")
        models[name] = model_from_checkpoint(saved).eval().to(device,memory_format=memory_format)
        frozen[name] = {"checkpoint_sha256": checksum(path), "config": saved["config"], "epoch": saved["epoch"] + 1}
    output_dir = args.artifacts / "results" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    pets = Pets(args.data, args.split)
    faces = Faces(args.data, args.split)
    pet_count = min(len(pets), args.limit) if args.limit else len(pets)
    face_count = min(len(faces), args.limit) if args.limit else len(faces)
    rows, classifier_true, classifier_pred, routing = [], [], [], []
    for start in range(0, pet_count, args.batch_size):
        items = [pets[i] for i in range(start, min(start + args.batch_size, pet_count))]
        image = torch.stack([item[0] for item in items]).to(device,memory_format=memory_format)
        target = torch.stack([item[1] for item in items]).to(device,memory_format=memory_format)
        label = torch.tensor([item[2] for item in items], device=device)
        logits = models["classifier"](image)
        probabilities = logits.softmax(1)
        predicted = probabilities.argmax(1)
        specialist_outputs = torch.stack([image] + [models[n](image) for n in CLASSES[1:]], 1)
        arange = torch.arange(len(items), device=device)
        soft, weights, _ = models["soft"](image)
        outputs = {"corrupted_baseline": image, "universal": models["universal"](image),
                   "oracle_routing": specialist_outputs[arange, label],
                   "predicted_routing": specialist_outputs[arange, predicted], "soft": soft}
        classifier_true.extend(label.cpu().tolist())
        classifier_pred.extend(predicted.cpu().tolist())
        for task, output in outputs.items():
            metrics = metric_values(output, target)
            for offset in range(len(items)):
                index = start + offset
                metadata = pets.rows[index]
                cfg = metadata["corruption"]
                rows.append({"index": index, "id": metadata["id"], "system": task, "corruption": cfg["kind"],
                             "severity": "clean" if cfg["kind"] == "clean" else cfg["level"],
                             "l1": metrics["l1"][offset], "psnr": metrics["psnr"][offset], "ssim": metrics["ssim"][offset],
                             "predicted": CLASSES[int(predicted[offset])], "routing_correct": bool(predicted[offset] == label[offset])})
        for offset in range(len(items)):
            metadata = pets.rows[start + offset]
            cfg = metadata["corruption"]
            routing.append({"id": metadata["id"], "corruption": cfg["kind"],
                            "severity": "clean" if cfg["kind"] == "clean" else cfg["level"],
                            **{f"weight_{name}": float(weights[offset, i]) for i, name in enumerate(CLASSES)},
                            **{f"probability_{name}": float(probabilities[offset, i]) for i, name in enumerate(CLASSES)}})
        if start % (args.batch_size * 20) == 0:
            print(f"Pets: {start}/{pet_count}", flush=True)
    write_csv(output_dir / "per_image.csv", rows)
    write_csv(output_dir / "routing.csv", routing)
    aggregates = []
    for system in outputs:
        for kind in CLASSES:
            for severity in (["clean"] if kind == "clean" else ("low", "medium", "high")):
                selected = [r for r in rows if r["system"] == system and r["corruption"] == kind and r["severity"] == severity]
                if selected:
                    aggregates.append({"system": system, "corruption": kind, "severity": severity, "n": len(selected),
                                       **{metric: float(np.mean([r[metric] for r in selected])) for metric in ("l1", "psnr", "ssim")}})
    write_csv(output_dir / "restoration_summary.csv", aggregates)
    matrix = confusion_matrix(classifier_true, classifier_pred, labels=list(range(4)), normalize="true")
    report = classification_report(classifier_true, classifier_pred, labels=list(range(4)), target_names=CLASSES, output_dict=True, zero_division=0)
    write_json(output_dir / "classifier.json", {"report": report, "normalized_confusion": matrix.tolist()})
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True", title="Normalized corruption confusion")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > .5 else "black")
    fig.tight_layout(); fig.savefig(output_dir / "confusion.png", dpi=160); plt.close(fig)
    route_groups = []
    for kind in CLASSES:
        for severity in (["clean"] if kind == "clean" else ("low", "medium", "high")):
            selected = [r for r in routing if r["corruption"] == kind and r["severity"] == severity]
            if selected:
                route_groups.append({"corruption": kind, "severity": severity, "n": len(selected),
                                     **{name: float(np.mean([r[f"weight_{name}"] for r in selected])) for name in CLASSES}})
    write_csv(output_dir / "routing_summary.csv", route_groups)
    fig, ax = plt.subplots(figsize=(6, 5))
    heat = np.array([[r[n] for n in CLASSES] for r in route_groups])
    ax.imshow(heat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(4), xticklabels=CLASSES, yticks=range(len(route_groups)),
           yticklabels=[r["corruption"] + " / " + r["severity"] for r in route_groups], title="Mean soft expert weights")
    fig.tight_layout(); fig.savefig(output_dir / "routing_heatmap.png", dpi=160); plt.close(fig)
    for task in ("universal", "predicted_routing", "soft"):
        task_rows = [r for r in rows if r["system"] == task]
        representative = representative_rows(task_rows)
        failures, seen = [], set()
        for row in sorted(task_rows, key=lambda r: r["psnr"]):
            if row["id"] not in seen:
                failures.append(row); seen.add(row["id"])
            if len(failures) == 4:
                break
        for label, selections in (("representative", representative), ("failures", failures)):
            examples = []
            for row in selections:
                image, target, _ = pets[row["index"]]
                batch = image[None].to(device)
                if task == "predicted_routing":
                    predicted = int(models["classifier"](batch).argmax(1))
                    output = batch if predicted == 0 else models[CLASSES[predicted]](batch)
                elif task == "soft":
                    output = models["soft"](batch)[0]
                else:
                    output = models[task](batch)
                output = output[0].cpu()
                description = f"{row['id']} | {row['corruption']} {row['severity']} | PSNR {row['psnr']:.2f}"
                examples.append((description, (target, image, output, (output - target).abs())))
            draw_grid(output_dir / f"{task}_{label}.png", examples, f"{task}: {label} ({args.split})")
            if label == "representative":
                for page in range(0, len(examples), 4):
                    draw_grid(output_dir / f"{task}_{label}_{page // 4 + 1}.png", examples[page:page + 4],
                              f"{task}: examples {page + 1}-{min(page + 4, len(examples))}")
            write_json(output_dir / f"{task}_{label}.json", selections)
    # Explicit evidence of confident versus distributed routing.
    soft_rows = [r for r in rows if r["system"] == "soft"]
    confident = sorted(range(len(routing)), key=lambda i: max(routing[i][f"weight_{n}"] for n in CLASSES), reverse=True)
    distributed = list(reversed(confident))
    write_json(output_dir / "routing_examples.json", {
        "dominant": [routing[i] for i in confident[:4]], "distributed": [routing[i] for i in distributed[:4]],
        "inactive_branches_below_one_percent": [n for n in CLASSES if np.mean([r[f"weight_{n}"] for r in routing]) < .01],
        "routing_error_examples": [r for r in rows if r["system"] == "predicted_routing" and not r["routing_correct"]][:12]})
    oracle_by_index = {r['index']: r for r in rows if r['system'] == 'oracle_routing'}
    routing_errors, routing_error_images, seen = [], [], set()
    for row in sorted((r for r in rows if r['system'] == 'predicted_routing' and not r['routing_correct']),
                      key=lambda r: r['psnr']):
        if row['id'] in seen:
            continue
        seen.add(row['id'])
        image, target, _ = pets[row['index']]
        batch = image[None].to(device)
        predicted_class = CLASSES.index(row['predicted'])
        restored = batch if predicted_class == 0 else models[row['predicted']](batch)
        restored = restored[0].cpu()
        oracle = oracle_by_index[row['index']]
        routing_errors.append({**row, 'oracle_psnr': oracle['psnr'],
                               'predicted_minus_oracle_psnr': row['psnr']-oracle['psnr']})
        description = f"{row['id']} | true {row['corruption']} -> {row['predicted']}"
        routing_error_images.append((description, (target,image,restored,(restored-target).abs())))
        if len(routing_errors) == 4:
            break
    write_json(output_dir/'routing_error_cases.json', routing_errors)
    if routing_error_images:
        draw_grid(output_dir/'routing_error_cases.png', routing_error_images, 'Classifier mistakes: hard routing')
    face_rows, face_examples = [], []
    face_example_counts = [0, 0, 0]
    for start in range(0, face_count, args.batch_size):
        items = [faces[i] for i in range(start, min(start + args.batch_size, face_count))]
        image = torch.stack([i[0] for i in items]).to(device)
        target = torch.stack([i[1] for i in items]).to(device)
        style = torch.tensor([i[2] for i in items], device=device)
        output = models["gan"](image, style)
        metrics = metric_values(output, target)
        for offset in range(len(items)):
            face_rows.append({"id": faces.rows[start + offset]["id"], "style": int(style[offset]),
                              **{metric: values[offset] for metric, values in metrics.items()}})
            if face_example_counts[int(style[offset])] < 4:
                face_example_counts[int(style[offset])] += 1
                face_examples.append((f"{faces.rows[start + offset]['id']} | Style {int(style[offset]) + 1}",
                                      (target[offset].cpu(), image[offset].cpu(), output[offset].cpu(), (output[offset] - target[offset]).abs().cpu())))
    write_csv(output_dir / "face_per_image.csv", face_rows)
    face_failures = sorted(face_rows, key=lambda row: row['psnr'])[:4]
    face_index = {row['id']: i for i,row in enumerate(faces.rows[:face_count])}
    face_failure_images = []
    for row in face_failures:
        photo, target, style = faces[face_index[row['id']]]
        restored = models['gan'](photo[None].to(device), torch.tensor([style],device=device))[0].cpu()
        face_failure_images.append((f"{row['id']} | Style {style+1} | PSNR {row['psnr']:.2f}",
                                    (target,photo,restored,(restored-target).abs())))
    write_json(output_dir/'face_failures.json', face_failures)
    draw_grid(output_dir/'face_failures.png',face_failure_images,f'Face-to-sketch failures ({args.split})')
    draw_grid(output_dir / "face_representative.png", face_examples, f"Face-to-sketch ({args.split})")
    for page in range(0, len(face_examples), 4):
        draw_grid(output_dir / f"face_representative_{page // 4 + 1}.png",
                  face_examples[page:page + 4],
                  f"Sketch examples {page + 1}-{min(page + 4, len(face_examples))}")
    face_summary = []
    for style in range(3):
        selected = [r for r in face_rows if r["style"] == style]
        if selected:
            face_summary.append({"style": style, "n": len(selected), **{m: float(np.mean([r[m] for r in selected])) for m in ("l1", "psnr", "ssim")}})
    write_csv(output_dir / "face_summary.csv", face_summary)
    # Same photograph under all three conditions; only its paired style has a reference.
    photo, _, _ = faces[0]
    variants = []
    for style in range(3):
        variants.append(models["gan"](photo[None].to(device), torch.tensor([style], device=device))[0].cpu())
    canvas = Image.new("RGB", (512, 155), "white")
    draw = ImageDraw.Draw(canvas)
    style_font = ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), 16)
    for i, (label, value) in enumerate(zip(("Photograph", "Style 1", "Style 2", "Style 3"), [photo] + variants)):
        draw.text((128 * i + 6, 5), label, fill="black", font=style_font)
        canvas.paste(Image.fromarray((value.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)), (128 * i, 25))
    canvas.save(output_dir / "style_comparison.png")
    summary = {"status": "evaluated", "split": args.split, "subset": bool(args.limit),
               'execution_memory_format': 'channels_last' if memory_format == torch.channels_last else 'contiguous',
               "pet_input_count": pet_count, "face_pair_count": face_count,
               "checkpoints": frozen, "restoration": aggregates, "classifier": report,
               "face": face_summary, "routing": route_groups}
    write_json(output_dir / "summary.json", summary)
    write_json(args.artifacts / "results" / "summary.json", summary)
    plots(args.artifacts, output_dir)
    print(f"Results saved to {output_dir}")


def plots(artifacts, output_dir):
    for path in sorted((artifacts / "checkpoints").glob("*/history.json")):
        rows = json.loads(path.read_text())
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="Training loss")
        ax.plot([r["epoch"] for r in rows], [r["validation_objective"] for r in rows], label="Validation objective")
        ax.set(xlabel="Epoch", ylabel="Loss / objective", title=path.parent.name)
        ax.legend(); fig.tight_layout(); fig.savefig(output_dir / f"curve_{path.parent.name}.png", dpi=160); plt.close(fig)
    for path in sorted((artifacts / "studies").glob("*.json")):
        study = json.loads(path.read_text())
        completed = [t for t in study["trials"] if t["state"] == "COMPLETE"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter([t["number"] for t in completed], [t["value"] for t in completed], color="#3a7552")
        ax.set(xlabel="Trial", ylabel="Validation objective", title=f"Optuna: {study['task']}")
        fig.tight_layout(); fig.savefig(output_dir / f"optuna_{study['task']}.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Debug subset, labelled as such; 0 means all")
    parser.add_argument("--device", default="auto")
    parser.add_argument('--cpu-channels-last',action='store_true',help='Use the benchmarked CPU tensor layout')
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
