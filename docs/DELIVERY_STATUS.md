# Project status — 4 October 2026

**State:** complete and runnable. All four workspaces, seven ONNX models, Optuna/MLflow records,
report and tests are in this repository.

* **Models.** Seven ONNX exports (universal, classifier, three specialists, soft mixture, GAN
  generator) pass checksum, finite-output and PyTorch-vs-ONNX checks (`artifacts/models/manifest.json`).
* **Application.** `docker compose up --build` serves the React frontend and FastAPI backend on
  http://localhost:8080. Container verification (`docs/container-verification.json`) covers all four
  workspaces, all three sketch styles, deterministic corruption and MLflow record access.
* **Results.** Final test evaluation of frozen checkpoints (36,690 pet inputs, 1,046 face pairs):
  classifier accuracy 96.85 %; restoration improves salt-and-pepper and occlusion PSNR by about 8–10 dB;
  blur is not improved on average; sketches are smooth. Full-validation reconstruction objective fell by
  65.2 % (universal), 68.2 % (predicted hard routing), 51.5 % (soft mixture) and 2.4 % (face-to-sketch)
  relative to the previous model generation, which is preserved outside the repository.
* **Tests.** 20 pytest tests pass (`python -m pytest -q`).
* **Training.** Colab pilots, then CPU training after the free GPU quota ended; Optuna budgets are small
  and are disclosed in the report.
* **Test split.** The official test split was also evaluated for two earlier model generations. All
  hyperparameter, architecture and checkpoint selection used validation data only.

See `README.md` for running instructions and `docs/SUBMISSION_CHECKLIST.md` for the requirement-to-evidence map.
