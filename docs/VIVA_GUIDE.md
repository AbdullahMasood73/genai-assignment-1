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
   Joint fine-tuning then updates them with a smaller learning rate. Softmax
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

## What actually went wrong

The original dense-bottleneck autoencoders collapsed to nearly constant outputs.
A normalization-only validation pilot did not fix this. A compressed spatial
latent retained coarse outlines and improved the validation objective, prompting
a separate repair run. This is experimental evidence, not proof that every new
output is good. Inspect the final measured tables and failure examples before
claiming successful denoising.

Two official test evaluations had already been seen before this improvement. All further improvement decisions use validation. Any final re-evaluation must
be disclosed; it cannot be described as an untouched test.
The initial frontend scaffold also preceded the original Stitch design; that
chronology is disclosed. Docker execution was verified through WSL. Public repository/video links must
be verified separately. Do not claim that a passing API or ONNX check proves
model quality or a container deployment.

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

