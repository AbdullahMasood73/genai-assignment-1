# Research and design decisions

This is the chronological research and experiment log. Design choices are hypotheses
tested on validation data, not claims of superiority; final test measurements and their
interpretation are in the report.

## Validation-led repair of the first run
The original universal model's selected validation objective was 0.32176, and its
saved validation examples were nearly constant. A three-epoch GroupNorm and
LeakyReLU pilot with the dense bottleneck reached 0.32742 and was rejected.
A second pilot kept the L1/SSIM loss and no-skip requirement, but compressed into
an 8x8x8 spatial latent (512 values versus 49,152 input values), expanded it with
1x1 convolutions, and decoded with bilinear resizing followed by convolutions.
Its best validation objective was 0.28057 at epoch two. The epoch-three grid
preserved coarse object outlines but had large color errors and blur. This is
evidence to justify further validation-led training, not a successful final model.
The full repair investigates latent sizes 256/512/1,024 and retains the other
required Optuna parameters. Its first candidate includes the validated pilot
settings. The original run and normalization-only pilot are preserved separately.
The first official test had already been evaluated before this iteration; that
chronology must be disclosed. No repair test measurements are used for tuning.

The layer implementations follow the official PyTorch APIs:
https://docs.pytorch.org/docs/2.8/generated/torch.nn.Upsample.html and
https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html.
The inference exporter passed six random/black/white checks at batch sizes one
and two for the spatial architecture (largest observed absolute error 2.21e-5).
These checks used synthetic weights and establish compatibility only.

## Restoration architecture
Vincent et al. establish corruption-to-clean learning as a representation-learning
objective. This project uses convolutional encoders and decoders with a dense
latent vector. This is not their stacked layerwise training method. A U-Net with
unrestricted skips could circumvent the assignment's bottleneck requirement, so
the restoration experts have no skips. The default 128-value vector compresses
49,152 input values; Optuna varies 64, 128 and 256. This aggressive compression
may lose texture and fine facial/pet details. Investigate this in failure examples.

Source: https://www.jmlr.org/papers/v11/vincent10a.html

## Reconstruction objective
L1 measures pixel differences; Gaussian-window SSIM measures local structural
agreement. The code implements the Wang et al. formulation with an 11x11,
sigma-1.5 Gaussian window, C1=0.01^2 and C2=0.03^2 on [0,1] RGB values. Valid
convolution excludes five border pixels. All systems use this same metric.
Alpha is tuned for training; validation ranking uses fixed alpha=0.8 so changing
alpha cannot directly alter the scoring scale. PSNR is computed per image, then
averaged (not computed from the dataset-wide mean MSE).

Source: https://ece.uwaterloo.ca/~z70wang/research/ssim/

## Hard and soft routing
The hard system has an identity bypass. An oracle supplies true labels for an
upper-bound routing comparison; operational inference uses classifier argmax.
The soft system evaluates every branch and blends outputs; it is not a sparse
conditional-compute layer. Shazeer et al. motivate investigating expert imbalance,
but this project uses the assignment's simpler squared deviation of mean weights
from 1/4. Balanced training batches make that target meaningful. Gate warm-up
freezes specialists; joint fine-tuning then unfreezes them. The implementation keeps one
learning rate (2.7e-4, searched in 1e-5 to 3e-4) for both stages; it is not lowered again
at unfreezing. Compare clean-image degradation, classifier errors, routing entropy,
inactive branches, and restoration quality rather than assuming soft is better.

Source: https://arxiv.org/abs/1701.06538

## Paired GAN and style
Isola et al. motivate paired conditional translation using a U-Net and a local
discriminator. Here a learned FS2K style embedding is concatenated as constant
feature maps into both networks. The project uses five downsampling layers at
128x128, four skip-connected upsampling layers, a sigmoid RGB output, and a
three-stride convolutional discriminator with a 38x38 receptive field. This is
a compact PatchGAN variant, not the paper's 70x70 configuration. No batch
normalization is used; batch-size-one inference remains simple, but GAN
stability and capacity require validation. Both members of a pair receive the
same horizontal flip. Changing style away from a photo's paired style is a
qualitative check only, since that target does not exist in the dataset.

Sources:
- https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html
- https://github.com/DengPingFan/FS2K

## Search and tracking
Optuna TPE handles categorical and continuous spaces. Median pruning is used
for individual-model studies; the shared specialist study scores the mean
validation objective of all three independently trained specialists. Default
short trials use a limited training/validation batch budget, disclosed in each
study JSON. Selected configurations are retrained from scratch on the entire
training split; test images are never used for selection. Limited trial count
does not justify claiming exhaustive optimization.

Sources:
- https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- https://mlflow.org/docs/latest/ml/tracking/

## Data sources and limitations
- Oxford: https://www.robots.ox.ac.uk/~vgg/data/pets/
- FS2K: https://github.com/DengPingFan/FS2K
- Official FS2K archive: https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view

Oxford official trainval is split 80/20 using NumPy seed 42; resulting membership
is persisted, not assumed identical to sklearn's RNG. FS2K official train is
stratified 85/15 with sklearn seed 42. Style comes from annotation, never from
the photo folder name. Test files may be downloaded/resized during preparation;
their pixels and labels are not consumed for training or tuning.

Outcome of this repair stage: seven exports, a full test evaluation and a Docker deployment check; images remained blurry with colour errors, which motivated the later improvement stage below. See DELIVERY_STATUS.md.

## Further validation-led improvement

The previous universal model compresses 49,152 input values to 512 latent values,
and its selected dropout is 0.260. The next pilot tests 4,096 latent values at
16x16 resolution (12x compression), lower dropout and local residual refinement.
There is no input/output residual addition and no cross-encoder-decoder skip:
all output information must pass through the compressed representation.
Residual blocks operate only within a feature stage. This is inspired by residual
autoencoder research, not a reproduction of that paper's JPEG-specific model.
GroupNorm computes per-group statistics in both training and evaluation; removing
it is a hypothesis about retaining absolute color information, not an established
cause until tested. Architecture choices use validation only. Earlier official
tests have already been seen, so this cannot restore an untouched-test protocol.

Sources: https://arxiv.org/abs/1903.06117 and
https://docs.pytorch.org/docs/2.8/generated/torch.nn.GroupNorm.html.

The lower-rate 16-epoch pilot reduced validation objective to 0.20738 at epoch 14, from 0.24516 previously, but its varied-image grid still showed substantial blur. A further pilot tests 8,192 latent values (6x compression) and BatchNorm with tracked inference statistics. BatchNorm uses population estimates during evaluation, unlike per-input GroupNorm; this is a validation-tested design alternative, not a claim that normalization alone guarantees fidelity. Source: https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html.

A learned convolution/PixelShuffle decoder is then compared with fixed bilinear resize-convolution. The first two epochs reduce full validation objective to 0.16458 (PSNR 19.18 dB), versus 0.24516 for the previous deployed universal model. These are intermediate pilot measurements, not final test scores. PixelShuffle rearranges learned channels into spatial subpixels; the bottleneck and absence of encoder-decoder bypasses remain unchanged. References: https://docs.pytorch.org/docs/2.8/generated/torch.nn.PixelShuffle.html and https://arxiv.org/abs/1609.05158.

The selected learned-upsampling pilot reached objective 0.14388949 at epoch 10, versus 0.24516189 for the preceding universal model (41.3% lower). It used ordinary subpixel initialization. Fresh final studies now use phase-matched ICNR initialization (Aitken et al., https://arxiv.org/abs/1707.02937) to address visible periodic artifacts; this initialization prevents phase differences initially but does not guarantee artifact-free trained outputs. The sketch model receives a fresh normalized U-Net/PatchGAN study and 100-epoch schedule rather than continuing the small prior generator. Final improvements remain subject to validation comparison and visual inspection.

At 2026-10-04 00:01 PKT the GPU runtime disconnected, and normal reconnection reported Colab free GPU usage limits. The selected pilot histories and images were downloaded, but its weights and incomplete new universal checkpoints were lost. The existing full preceding delivery remained safe. CPU benchmarking measured about 0.68-0.97 seconds per training batch of 16 and 0.248 seconds per validation batch. A fresh CPU run uses the saved real pilot evidence to seed studies. Before candidate test evaluation, its fixed deadline-aware schedule was reduced to 24 universal, 20 per specialist, 12 joint-mixture, and 60 GAN epochs; new studies use three trials, three short epochs, and 40/40 train/validation batches. Local data manifests match prior Colab fingerprints after CRLF-to-LF normalization; no split or corruption configuration changed. The interrupted CPU study with the earlier GPU budget is preserved separately and is not represented as a completed search.

The first fresh CPU trial (three short epochs, 40/40 batches) reached validation-sample objective 0.14329160 and PSNR 20.5008 dB. This sample score is not comparable with the complete-split pilot score without full evaluation. A 12-image random full-validation grid was inspected: noise is substantially reduced and object outlines/colors remain, but eyes, fur, background texture and clean-image detail are still blurred. Large occluded face regions are not reconstructed accurately. These observations support longer training and meaningful failure analysis, not a claim of uniformly successful restoration.

Before final improved evaluation, the figure routine was updated for print legibility: 16-pixel labels, four-example report panels, and a fixed first-four-per-style rule for sketch representative images. This changes only illustrative selection and formatting, not numerical measurements. Validation now reuses the already computed SSIM value instead of computing it twice; an exact tensor-equality check confirmed the same objective, and all 20 pipeline tests passed afterwards.

A disposable CPU-layout benchmark on random inputs compared contiguous and channels-last tensors while real training remained active. Batch-16 median training times were 1.966 versus 1.400 seconds, and validation times 0.670 versus 0.492 seconds; initial outputs differed by at most 5.96e-8. This is execution-performance evidence, not dataset-quality evidence. CPU layout changes preserve tensor shapes and model/loss definitions, but floating-point accumulation can differ. Source: https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/. Checkpoint resume/export validation precedes adoption.

At the 02:42 PKT follow-up, universal training had completed23 of24 full epochs. Best complete-validation objective0.0852753 at epoch22 (65.2% below the preceding0.2451619); PSNR24.5325dB and SSIM0.74082. The same12 fixed seed-42 validation inputs were visually reviewed again: substantially less salt noise, preserved overall animal shapes/colors, but remaining fur/eye/background blur, clean-image detail loss and smooth inaccurate fill in large occlusions. These are interim universal results, not full-pipeline promotion or test claims. CPU epochs now take about284seconds rather than the earlier430–490seconds.

The completed20-epoch salt specialist selected epoch20: full salt-validation objective0.0675619, PSNR26.2828dB, SSIM0.80075. Preceding oracle-routing salt rows have weighted objective0.2380316 on the same three fixed severity sets, a71.6% reduction. Twelve seeded salt-only images show strong noise removal but retained fine-detail blur. The inspection helper now automatically filters a specialist to its own corruption; universal inspection remains across all conditions. No test data was used in this comparison.

At05:31PKT the blur specialist completed20 epochs; selected epoch18 objective0.06829934, PSNR26.3224dB and SSIM0.79672. Preceding oracle blur rows have fixed-objective weighted mean0.23809641 on the same validation inputs (71.3% reduction). Twelve seeded blur-only images show preserved colors/shapes and uneven edge recovery, remaining fine-detail loss and occasional edge/color artifacts. This comparison does not establish superiority to the corrupted-input baseline in every severity group. Full pipeline validation and test remain pending.
