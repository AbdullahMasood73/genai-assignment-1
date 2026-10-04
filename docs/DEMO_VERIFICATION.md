# Application verification

`docs/container-verification.json` records a real Docker Compose run at http://127.0.0.1:8080:
health ready for all seven models, reproducible corruption for salt/blur/occlusion, successful
universal, hard-routing, soft-mixture and face-to-sketch (styles 1–3) requests with valid PNG output,
and 50 MLflow runs visible through the application. `docs/local-verification.json` records
checksum, finite-output and input-sensitivity checks for the ONNX models. These are functional
checks, not quality or throughput benchmarks. The recording plan for the demonstration video is in
`docs/DEMO_SCRIPT.md`.
