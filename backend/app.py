"""CPU ONNX API. Untrained/missing models return 503; there is no fake inference."""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
import warnings
import sqlite3
from contextlib import closing

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from restoration.corruptions import CLASSES, apply, config

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE / "artifacts/models")))
MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000
app = FastAPI(title="Restoration Lab", version="1.0", description="Four generative imaging workspaces")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def model_entries():
    path = MODEL_DIR / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())["models"]
    except (ValueError, KeyError):
        return {}


def registered_model(name):
    entry = model_entries().get(name)
    if not entry or not entry.get("verified") or entry.get("synthetic"):
        raise HTTPException(503, f"The {name} model is not trained and verified yet. Complete training and import the model bundle.")
    # Resolve fixed names, never arbitrary filenames from a client/manifest.
    path = MODEL_DIR / f"{name}.onnx"
    if not path.exists():
        raise HTTPException(503, f"Missing {name}.onnx")
    return path, entry


@lru_cache(maxsize=7)
def verified_session(name, path_string, expected_hash, modified):
    path = Path(path_string)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise HTTPException(503, f"Model checksum mismatch: {name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    try:
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as error:
        raise HTTPException(503, f"Cannot load {name} model") from error


def session(name):
    path, entry = registered_model(name)
    return verified_session(name, str(path), entry.get("sha256", ""), path.stat().st_mtime_ns)


def run(name, image, style=None):
    feed = {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)}
    if style is not None:
        feed["style"] = np.array([style], dtype=np.int64)
    return session(name).run(None, feed)


def encode(image):
    buffer = BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


async def decode(file):
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP"):
                    raise HTTPException(415, "Use a PNG, JPEG, or WebP image.")
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                original_size = list(image.size)
                image = image.resize((128, 128), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.float32) / 255
        return array, original_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise HTTPException(400, "The uploaded image could not be read safely.") from error


@app.get("/api/health")
def health():
    status = {}
    for name in ("universal", "classifier", "salt", "blur", "occlusion", "soft", "gan"):
        try:
            session(name)
            status[name] = "ready"
        except HTTPException:
            status[name] = "pending"
    workspaces = {"universal": status["universal"] == "ready",
                  "hard": all(status[n] == "ready" for n in ("classifier", "salt", "blur", "occlusion")),
                  "soft": status["soft"] == "ready", "gan": status["gan"] == "ready"}
    return {"status": "ready" if all(workspaces.values()) else "awaiting_models", "models": status,
            "workspaces": workspaces, "image_size": 128, "provider": "CPUExecutionProvider"}


@app.get("/api/samples")
def samples():
    folder = BASE / "backend" / "samples"
    return [{"id": path.name, "url": f"/samples/{path.name}"} for path in sorted(folder.glob("*.png"))[:8]]


@app.get("/api/experiments")
def experiments():
    path = MODEL_DIR.parent / "results" / "summary.json"
    if not path.exists():
        return {"status": "pending", "message": "Evaluation results will appear after training."}
    result = json.loads(path.read_text())
    database = MODEL_DIR.parent / "tracking.db"
    records = []
    if database.is_file():
        try:
            # Deployment reads a completed experiment snapshot, never mutates it.
            with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                records = [dict(row) for row in connection.execute("""
                    SELECT r.run_uuid AS id, r.name, r.status,
                           MAX(CASE WHEN m.key='epoch' THEN m.value END) AS epoch,
                           MAX(CASE WHEN m.key='validation_objective' THEN m.value END) AS validation_objective
                    FROM runs r LEFT JOIN latest_metrics m ON r.run_uuid=m.run_uuid
                    WHERE r.lifecycle_stage='active'
                    GROUP BY r.run_uuid ORDER BY r.start_time DESC LIMIT 50
                """)]
        except sqlite3.Error:
            records = []
    result["tracking"] = {"source": "MLflow", "runs": records}
    return result


@app.post("/api/corrupt")
async def corrupt(file: UploadFile = File(...), corruption: str = Form("salt"),
                  severity: str = Form("medium"), seed: int = Form(42)):
    image, dimensions = await decode(file)
    if corruption not in CLASSES or severity not in ("low", "medium", "high") or not 0 <= seed < 2**31:
        raise HTTPException(422, "Invalid corruption, severity, or seed.")
    cfg = config(corruption, seed, None if corruption == "clean" else severity)
    output = apply(image, cfg)
    return {"input": encode(image), "output": encode(output), "settings": cfg, "original_size": dimensions}


async def restore(kind, file, corruption, severity, seed):
    image, dimensions = await decode(file)
    if corruption not in ("none",) + CLASSES or severity not in ("low", "medium", "high") or not 0 <= seed < 2**31:
        raise HTTPException(422, "Invalid corruption settings.")
    cfg = None if corruption == "none" else config(corruption, seed, None if corruption == "clean" else severity)
    corrupted = apply(image, cfg) if cfg else image
    # Load outside timer: inference timing excludes first-time disk/session initialization.
    required = ["classifier", "salt", "blur", "occlusion"] if kind == "hard" else [kind]
    for name in required:
        session(name)
    started = time.perf_counter()
    extra = {}
    if kind == "hard":
        logits = run("classifier", corrupted)[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        predicted = int(probabilities.argmax())
        output = corrupted if predicted == 0 else run(CLASSES[predicted], corrupted)[0][0].transpose(1, 2, 0)
        extra = {"probabilities": probabilities.tolist(), "predicted": CLASSES[predicted],
                 "expert": "identity bypass" if predicted == 0 else CLASSES[predicted]}
    elif kind == "soft":
        restored, weights = run("soft", corrupted)
        output = restored[0].transpose(1, 2, 0)
        extra = {"weights": weights[0].tolist()}
    else:
        output = run("universal", corrupted)[0][0].transpose(1, 2, 0)
    milliseconds = (time.perf_counter() - started) * 1000
    result = {"input": encode(corrupted), "output": encode(output), "inference_ms": round(milliseconds, 2),
              "settings": cfg, "original_size": dimensions, "resolution": [128, 128], **extra}
    if cfg:
        result.update(target=encode(image), error_map=encode(np.abs(output - image)))
    return result


@app.post("/api/universal-restoration")
async def universal(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42)):
    return await restore("universal", file, corruption, severity, seed)


@app.post("/api/hard-routing")
async def hard(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42)):
    return await restore("hard", file, corruption, severity, seed)


@app.post("/api/soft-mixture")
async def soft(file: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42)):
    return await restore("soft", file, corruption, severity, seed)


@app.post("/api/face-to-sketch")
async def sketch(file: UploadFile = File(...), style: int = Form(0)):
    if style not in (0, 1, 2):
        raise HTTPException(422, "Select Style 1, Style 2, or Style 3.")
    image, dimensions = await decode(file)
    session("gan")
    started = time.perf_counter()
    output = run("gan", image, style)[0][0].transpose(1, 2, 0)
    milliseconds = (time.perf_counter() - started) * 1000
    return {"input": encode(image), "output": encode(output), "style": style,
            "inference_ms": round(milliseconds, 2), "original_size": dimensions, "resolution": [128, 128]}


samples_dir = BASE / "backend" / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")
