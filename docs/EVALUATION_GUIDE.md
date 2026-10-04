# Implementation walkthrough for live evaluation

1. `prepare.py` reads official split annotations and writes RGB 128x128 images,
   split membership, and deterministic corruption manifests. Dataset categories
   are not restoration labels; corruption type is generated at load time.
2. `data.py` creates fresh random corruptions on each training load. Classifier
   and gate batches contain equal counts of all four corruption conditions.
3. `models.py` contains compressed autoencoder variants (the improvement uses a
   16×16 latent grid with local residual refinement, tracked normalization and
   learned upsampling), the four-class classifier,
   soft gate, style-conditioned U-Net, and conditional local discriminator.
4. `train.py` searches on validation, retrains selected models, saves optimizer
   and RNG states, and logs real experiments to MLflow. The gate starts from the
   classifier and experts start from independently trained specialists.
5. `evaluate.py` freezes checkpoint hashes before final testing and compares
   baseline/universal/oracle/predicted/soft by corruption and severity. It writes
   per-image evidence, not only averages.
6. `export.py` checks PyTorch versus ONNX for random/black/white inputs, batch
   sizes 1 and 2, and all three sketch styles. The backend verifies model hashes.
7. `backend/app.py` validates uploads, applies optional corruption and runs CPU
   ONNX inference. Hard clean routing bypasses restoration; soft inference runs
   all four branches. The GAN discriminator is never deployed.

Be ready to answer:
- Why is the compressed latent a real bottleneck? Why omit restoration skip connections?
- Why combine L1 with SSIM? Why keep the validation weighting fixed across trials?
- What can a classifier mistake do to specialist restoration?
- Why does the soft mixture require a temperature and balance regularizer?
- What changes during gate warm-up versus joint fine-tuning?
- How is a style embedding different from a UI label?
- Why must paired spatial augmentations be identical?
- Why can PSNR/SSIM miss plausible sketch details?
- How do you verify that test data never drives hyperparameter selection?
- What evidence supports a claim that one method is better?

The assignment requires the student to understand the implementation. This guide
supports preparation; it cannot substitute for practicing these explanations.
