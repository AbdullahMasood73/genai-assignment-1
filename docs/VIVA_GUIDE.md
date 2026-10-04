# Explaining the submitted system

The app has four workspaces, a React frontend and a FastAPI backend. Inference
uses exported ONNX models on the CPU. Exploratory pilots used PyTorch on a free Colab T4. Final improved models use
the laptop CPU after Colab reached its free GPU limit.
Training and serving are separate: the laptop needs only the inference models.

## Four tasks

1. **Universal:** one autoencoder maps any supported input condition to the clean
   image. Its encoder reduces 128×128 to 16×16. The final architecture compresses
   channels to a spatial latent containing 2,048, 4,096 or 8,192 values, selected by
   Optuna. The original input contains 49,152 values. No encoder skip bypasses
   the bottleneck. The decoder upsamples back to RGB at 128×128.
2. **Hard routing:** a four-class classifier predicts clean, salt noise, blur or
   occlusion. Clean uses the input unchanged. Other predictions select one of
   three independently trained specialists. Oracle evaluation uses the known
   corruption label, so its difference from predicted routing isolates routing
   mistakes. Good classification does not guarantee good reconstruction.
3. **Soft mixture:** the pretrained classifier initializes the gate; the trained
   specialists initialize the experts. A two-epoch warm-up freezes the experts.
   Joint fine-tuning then updates them; the same learning rate (2.7e-4) is kept —
   it is not lowered at unfreezing, which the assignment suggests. Softmax
   weights blend the identity and all three experts, so gradients can flow through
   the mixture. A large identity weight can leave noise unchanged.
4. **Face-to-sketch:** a paired conditional GAN learns a photograph plus one of
   three categorical styles. A learned embedding enters both the U-Net generator
   and PatchGAN discriminator. The new candidate adds normalization and learned upsampling to improve stability
   and fine detail. Only the generator is exported for inference.
   Paired flips are identical for photograph and sketch.

## Losses and selection

- Autoencoders use `alpha × L1 + (1 − alpha) × (1 − SSIM)`; Optuna tunes alpha.
  Validation ranking fixes alpha at 0.8 to compare configurations fairly.
- Classification uses cross-entropy. Runtime labels are known when corruption
  is generated; classifier and gate batches are balanced across the four classes.
- The mixture adds classification and branch-balance terms to reconstruction.
  Temperature changes how concentrated its softmax weights are.
- The GAN uses adversarial loss plus weighted L1 reconstruction. Generator and
  discriminator losses are logged separately.
- Best checkpoints are selected using validation, rather than the last epoch.
  Short Optuna trials save compute but can favor configurations that learn quickly.

## Data and metrics

Oxford Pets uses official test membership and a seed-42 development split:
2,944 training, 736 validation and 3,669 test images. Each test image has one clean
and nine deterministic corruption conditions, totaling 36,690 restoration inputs.
FS2K has 899 training, 159 validation and 1,046 test pairs. Its validation split
is stratified by style. Training corruptions are generated dynamically; evaluation
manifests retain seeds, masks and blur settings.

L1 measures pixel error. PSNR expresses per-image squared reconstruction error
on a logarithmic scale. SSIM measures local structure and contrast. Compare
results by corruption and severity, including the corrupted-input baseline.
Average scores can hide clean-image degradation or severe occlusion failures.

## Development history and caveats

The first dense-bottleneck autoencoders collapsed to nearly constant outputs, and a
normalization-only pilot did not fix this. A compressed 8x8 spatial latent kept coarse outlines;
a 16x16 latent with local residual blocks improved further; the final design (8,192-value latent,
BatchNorm, PixelShuffle upsampling) reached a universal validation objective of 0.085. These pilots
are sequential changes, not controlled ablations.

State these caveats plainly if asked:
- The official test split was also evaluated for two earlier model generations. All selection used validation data.
- Optuna budgets are small (3-4 short trials per study); in three studies the seeded starting configuration won.
- Blur is not improved on average (the compressed latent caps fine detail); the soft mixture comes closest.
- Sketches are smooth and lack hatching; style 3 has only 46 test pairs.
- The soft mixture keeps one learning rate for warm-up and joint fine-tuning (no second reduction).
- Final models were trained on a CPU after the free Colab GPU quota ended.
- A passing API or ONNX check proves the pipeline runs, not that the model is good.

## Where to change things (likely live-modification requests)

| Request | Where |
|---|---|
| Blur kernel/sigma, salt probability, occlusion count/area, severities | `config()` and `rectangles()` in `restoration/corruptions.py` |
| L1/SSIM weight alpha | `alpha` in the config JSON; `reconstruction()` in `restoration/models.py` |
| Latent (bottleneck) size or channel width | `latent` and `base` in the config (`latent` is a multiple of 256 in detail mode); `Autoencoder` in `restoration/models.py` |
| Soft temperature | `temperature` in the config; `SoftMixture.forward` in `restoration/models.py` |
| Balance loss | `balance = (weights.mean(0) - 0.25).square().sum()` in `restoration/train.py` (soft branch, around line 231) |
| Gate warm-up length | `--warmup-epochs` (default 2) in `restoration/train.py` |
| Add or remove an expert | `SoftMixture` and `component_models` (models.py/train.py), `restoration/export.py`, `backend/app.py` |
| GAN losses, L1 weight, style embedding size | GAN branch of `train_task` in `restoration/train.py` (around lines 208-225); `embedding` in the config; `Generator`/`Discriminator` in models.py |
| Paired augmentation | `Faces.__getitem__` in `restoration/data.py` |
| Image size | `read_rgb` in `restoration/corruptions.py`, `restoration/prepare.py`, `decode` in `backend/app.py` (models must be retrained) |
| New API field or endpoint | `restore()` / route functions in `backend/app.py`; `request()` in `frontend/src/App.tsx` |

## Useful code locations

`restoration/corruptions.py`: noise, blur and masks.
`restoration/data.py`: dynamic loading and balanced batches.
`restoration/models.py`: networks, SSIM and reconstruction loss.
`restoration/train.py`: tuning, optimization, warm-up and checkpoints.
`restoration/evaluate.py`: frozen-model evaluation and figures.
`restoration/export.py`: ONNX parity checks across inputs and batch sizes.
`backend/app.py`: validated uploads, model loading and inference.
`frontend/src/App.tsx`: four workspaces, weights, downloads and experiment records.
`compose.yaml`: intended container startup via `docker compose up --build`.

## Current improvement evidence

The selected 12-epoch learned-upsampling pilot reached validation objective
0.143889 at epoch 10, versus 0.245162 for the preceding universal model, about
41.3% lower. This is a measured pilot result, not a promised final grade or a
claim that all tasks improved. Earlier high-learning-rate and bilinear pilots
were weaker. Final full validation and visual review selected the improved models: universal objective 0.085275, hard routing 0.071003, soft mixture 0.064218 and GAN 0.185323. The GAN remains blurry and style 1 slightly regressed. Two historical test evaluations were already observed; the final frozen candidates underwent one disclosed third test.

Be able to explain why the latent remains a bottleneck despite local residual
blocks, how oracle routing differs from predicted routing, why identity can
preserve corruption, and why numerical export parity does not establish image
quality. Practice a new image in each workspace before the viva.

