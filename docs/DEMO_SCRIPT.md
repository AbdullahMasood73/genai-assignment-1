# Demonstration recording plan (target 6:00, allowed 5–7 minutes)

Record the real screen (OBS Studio or the Windows Game Bar, Win+G), speak live, upload to YouTube
(unlisted is fine) and put only the link in the report and README.

## Starting the app on the author's Windows laptop (Docker runs inside WSL)

This laptop has no Docker Desktop; Docker Engine runs inside the Ubuntu WSL distro with a temporary socket.
If the engine is not running (for example after a reboot or `wsl --shutdown`), start it first and leave that
window open:

```powershell
wsl -d Ubuntu-22.04 -u root bash "/mnt/c/Users/PC/OneDrive - FAST National University/Desktop/Codex/genai-assignment/scripts/start_wsl_docker.sh"
```

Then, in a second terminal, show the startup that the video needs:

```powershell
wsl -d Ubuntu-22.04 -u root
export DOCKER_HOST=unix:///tmp/restoration-lab-docker.sock
cd /tmp/restoration-lab        # a clone of this repository
docker compose down
docker compose up --build      # open http://localhost:8080 in the browser
```

On any machine with Docker Desktop or Docker Engine the same two commands inside the repository folder are
`docker compose up --build` (no `DOCKER_HOST`).

Before recording: run `docker compose down`, close other windows, open a terminal in the repository,
and keep a few unseen photographs (a pet photo, a face photo) ready.

| Time | Show | Say |
|---|---|---|
| 0:00–0:40 | Terminal: `docker compose up --build` (or `up` if already built), then http://localhost:8080 and `curl http://localhost:8080/api/health` | One command starts frontend and backend; models are ONNX files in `artifacts/models`; all four workspaces report ready |
| 0:40–1:40 | **Universal Restoration**: upload a pet image → apply salt-and-pepper (high) → Restore; then blur and occlusion; open the clean target and error map; point at settings and inference time; **Download result** | Runtime corruption with a seed; one autoencoder with a compressed latent; error map shows where it fails |
| 1:40–2:30 | **Hard-Routed Restoration**: same image, show four probabilities, predicted class, selected expert; try a clean image to show the identity bypass | Classifier argmax selects one specialist; wrong labels can pick the wrong expert |
| 2:30–3:20 | **Soft Mixture**: same input; show the four weight bars; compare with the hard result | Jointly trained gate blends branches; weights are continuous; low-severity blur shares weight with identity |
| 3:20–4:20 | **Face-to-Sketch**: upload a photo (or use the webcam capture button), generate Style 1, 2 and 3, **Download result** | Learned style embedding conditions generator and discriminator; sketches are smooth — a known limitation |
| 4:20–5:20 | **Experiments** panel: test metrics and the MLflow run table; then `mlflow ui --backend-store-uri sqlite:///artifacts/tracking.db --port 5000` in the browser; show `report/report.pdf` figures (confusion matrix, routing heatmap, Optuna plots) | Real Optuna trials and MLflow runs; 96.8 % classifier accuracy; blur limitation and why |
| 5:20–6:00 | Restart the app (`docker compose restart`), run one more unseen image, finish | CPU ONNX inference takes tens of milliseconds; the whole system starts from one command |

Checklist of items the assignment says the video must show: application startup · image upload ·
runtime corruption · universal restoration · hard routing · soft expert weights · face-to-sketch
generation · result downloading · experiment-tracking records.
