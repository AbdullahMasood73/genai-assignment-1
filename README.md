# Restoration Lab

Generative AI Assignment 1 — Muhammad Abdullah Masood (23I-0756)

One browser application, four workspaces, one `docker compose up`:

| # | Workspace | What it does |
|---|-----------|--------------|
| 1 | **Universal Restoration** | One compressed denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images. |
| 2 | **Hard-Routed Restoration** | A 4-class corruption classifier picks one of three specialist autoencoders (clean images bypass restoration). |
| 3 | **Soft Mixture-of-Experts Restoration** | A jointly trained gate blends an identity branch and the three experts with softmax weights. |
| 4 | **Face-to-Sketch Generator** | Style-conditioned U-Net / PatchGAN producing an FS2K-style sketch from a photograph or webcam capture. |

Tasks 1–3 use the Oxford-IIIT Pet dataset; Task 4 uses FS2K. All images are 128 × 128 RGB.

| Resource | Where |
|----------|-------|
| GitHub repository | https://github.com/AbdullahMasood73/genai-assignment-1 |
| Technical report (IEEE, LaTeX) | `report/report.pdf` (source: `report/report.tex`, `report/template.tex`) |
| Demonstration video (YouTube) | https://youtu.be/iDYYIRUBiP0 |
| ONNX models (all 7, ~38 MB) | `artifacts/models/` — included in this repository, SHA-256 in `manifest.json` |
| Optuna studies | `experiments/final_run/` (`optuna.db`, `studies/*.json`) and `artifacts/studies/` |
| MLflow records | `artifacts/tracking.db` + `artifacts/mlflow_artifacts/` |
| Google Stitch design | https://stitch.withgoogle.com/projects/16111510614845497849 and exports in `docs/stitch-evidence/` |
| Requirement-by-requirement evidence map | `docs/SUBMISSION_CHECKLIST.md` |

## Quick start (evaluator path)

Prerequisite: Docker with Compose. No GPU, Python or Node installation is needed.

```sh
git clone https://github.com/AbdullahMasood73/genai-assignment-1.git
cd genai-assignment-1
docker compose up --build
```

Open **http://localhost:8080**. The first build downloads base images and takes a few
minutes. The backend serves the ONNX models that are committed under `artifacts/models/`;
the status indicator at the top of the page reads "Service connected" once the API is up. Check that all four workspaces are ready with:

```sh
curl http://localhost:8080/api/health
```

Stop with `Ctrl+C`, then `docker compose down`. The site is bound to `127.0.0.1:8080`
(edit `compose.yaml` to change it).

Using the app:
1. Pick a workspace in the left rail.
2. Upload an image or choose a clean sample; for Tasks 1–3 choose a corruption, severity and
   seed (or upload an already corrupted image with *no corruption* selected).
3. **Restore image** shows input, output, inference time, applied settings, and (when a clean
   reference exists) the target and absolute-error map. Hard routing shows the four classifier
   probabilities, predicted class and selected expert; soft mixture shows the four gate weights.
4. Face-to-Sketch: upload or capture a photo, choose Style 1/2/3, generate, then download.
5. *Experiments* shows test metrics and the MLflow run table.

## Results (official test split, frozen checkpoints)

Mean PSNR (dB) over the three severities; 36,690 pet inputs (3,669 images × 10 conditions) and
1,046 face pairs. Per-severity tables, SSIM, error maps, failure cases and the routing heatmap
are in the report and in `artifacts/results/test/`.

| System | Salt & pepper | Blur | Occlusion | Clean (SSIM) |
|--------|--------------:|-----:|----------:|-------------:|
| Unrestored input | 16.4 | **27.8** | 13.5 | 1.000 |
| Universal (Task 1) | 24.7 | 25.6 | 21.8 | 0.822 |
| Oracle routing (Task 2) | 26.0 | 26.0 | 21.8 | 1.000 |
| Predicted routing (Task 2) | 25.9 | 26.2 | 21.7 | 0.971 |
| Soft mixture (Task 3) | **26.1** | 27.4 | 20.4 | 0.995 |

* Classifier: 96.8 % accuracy, macro F1 0.950 (clean recall 0.83 — clean is confused with low-severity blur).
* Restoration clearly helps for salt-and-pepper (+8 to +10 dB) and occlusion (+8 dB PSNR).
* **Blur is not improved on average**: the compressed latent limits fine detail, so the unrestored
  blurred image has higher PSNR than most restored outputs; the soft mixture comes closest.
* Face-to-sketch: mean SSIM 0.455; sketches are smooth and lack hatching (style 3 has only 46 test pairs).
* Inference on CPU: about 5–30 ms per request.

These limitations are analysed in the report (Discussion and Limitations sections).

## Repository layout

```
backend/      FastAPI service (ONNX Runtime, upload validation, corruption, timing) + Dockerfile
frontend/     React + TypeScript + Tailwind CSS (Vite) + Nginx Dockerfile
restoration/  corruptions, data pipeline, models, training (Optuna + MLflow), evaluation, ONNX export
scripts/      data checks, report builder, packaging and verification helpers
tests/        20 automated tests (pytest)
artifacts/    models/ (ONNX), results/ (test metrics, figures), studies/, tracking.db, mlflow_artifacts/
experiments/  Optuna databases/study summaries and pilot records for each run
notebooks/    Colab training notebook (embeds the source)
report/       IEEE LaTeX report, generated figures, compiled PDF
docs/         research notes, evaluation guide, screenshots, Stitch exports, verification records
```

## Reproducing the pipeline (optional)

Datasets and large checkpoints (about 700 MB) are **not** in the repository; the ONNX models and
results are. To retrain from scratch:

```sh
python -m venv .venv
# activate the environment with your shell's usual command
python -m pip install -r requirements-train.txt

# 1. Download/prepare data (official splits, seed 42, corruption manifests)
python -m restoration.prepare --dataset all

# 2. Optuna search + final training for every task, logged to MLflow
python -m restoration.train --task universal   --artifacts artifacts_reproduction --detail --detail-bn --detail-shuffle --epochs 24 --trials 3 --trial-validation-batches 40
python -m restoration.train --task classifier  --artifacts artifacts_reproduction --epochs 12 --trials 3 --trial-validation-batches 40
python -m restoration.train --task specialists --artifacts artifacts_reproduction --detail --detail-bn --detail-shuffle --epochs 20 --trials 3 --trial-validation-batches 40
python -m restoration.train --task soft        --artifacts artifacts_reproduction --epochs 12 --trials 3 --trial-validation-batches 40
python -m restoration.train --task gan         --artifacts artifacts_reproduction --gan-refined --epochs 60 --trials 3 --trial-validation-batches 40

# 3. Evaluate (validation first; run --split test only after freezing the models)
python -m restoration.evaluate --split validation
python -m restoration.evaluate --split test

# 4. Export to ONNX with PyTorch-vs-ONNX consistency checks, then build the report
python -m restoration.export
python scripts/build_report.py
```

`--mode tune|train|both` selects stages; `--backup <dir>` snapshots checkpoints and databases.
The notebook `notebooks/Assignment1_Train.ipynb` runs the same pipeline on a Colab GPU.
The delivered models were produced by short Colab pilots followed by CPU training; a fresh
reproduction does not claim identical trials or weights.

Data details: Oxford — 2,944 train / 736 validation / 3,669 test; FS2K — 899 train / 159
validation (stratified by style) / 1,046 test. Hashes are in `docs/data_preparation.json`.

Inspect experiment tracking:

```sh
pip install mlflow
mlflow ui --backend-store-uri sqlite:///artifacts/tracking.db --port 5000
```

## Local development without Docker

```sh
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
cd frontend && npm install && npm run dev      # http://127.0.0.1:5173 (proxies /api)
```

## Tests

```sh
python -m pytest -q          # 20 tests
cd frontend && npm run build
```

Tests cover deterministic corruptions, balanced batches, paired augmentation, SSIM, the compressed
latent, gradients through all mixture branches, style conditioning in generator and discriminator,
upload validation, checkpoint resume, and every ONNX graph. Synthetic-data tests check mechanics only.

## Limitations (summary)

Small Optuna budgets (3–4 short trials per study; in three studies the seeded starting configuration
won), no blur improvement on average, clean/low-blur confusion in the classifier and gate, smooth
sketches, imbalanced FS2K test styles (619/381/46), and CPU-only final training. The test split was also
evaluated for two earlier model generations during development; all selection used validation data.
See `report/report.pdf` for the full analysis.

## AI assistance

OpenAI Codex, Anthropic Claude (Claude Code) and Google Stitch were used; tools, tasks and
verification are listed in Appendix A of the report. Walkthrough notes for explaining the system are in
`docs/EVALUATION_GUIDE.md` and `docs/VIVA_GUIDE.md`.
