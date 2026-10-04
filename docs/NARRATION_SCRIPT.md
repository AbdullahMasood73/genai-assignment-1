# Narration script (about 6 minutes, ~850 words)

Read each chapter in your own words or as written. Record **one file per chapter** (Windows "Voice
Recorder"), speak at a calm pace, leave about one second of silence at the start and end of each file,
and name them `ch1`, `ch2`, … `ch7`. The picture will be timed to your audio, so the length of each file
decides how long that part of the screen recording lasts.

## ch1 — Introduction and startup (~55 s)
Hello, I am Muhammad Abdullah Masood, roll number 23I-0756. This is my Generative AI Assignment 1:
Restoration Lab. It is one browser application with four workspaces: universal restoration, hard-routed
restoration, a soft mixture of experts, and a face-to-sketch generator. Tasks one to three use the Oxford-IIIT
Pet dataset and task four uses FS2K. The whole system starts with one command, docker compose up --build. It
starts a React and Tailwind frontend behind Nginx, and a FastAPI backend that runs seven ONNX models on the CPU.
The health check confirms that all seven models are loaded and all four workspaces are ready.

## ch2 — Universal Restoration (~65 s)
The first workspace is Universal Restoration. I upload an image, choose a corruption and a severity, and the
backend applies it at runtime with a fixed seed, so it is reproducible. Here is salt-and-pepper noise at high
severity. One autoencoder with a compressed latent restores the image. The app shows the input, the restored
output, the corruption settings, and the inference time, which is about fifteen milliseconds on a CPU. Opening
the comparison shows the clean target and the absolute error map. Now blur, and now occlusion. The restoration
fills the black masks with plausible colors but cannot recover fine texture. I can download the result as a PNG.

## ch3 — Hard-Routed Restoration (~55 s)
The second workspace is hard routing. A four-class classifier looks at the input and outputs probabilities for
clean, salt-and-pepper, blur and occlusion. The argmax picks exactly one specialist autoencoder, and the app
shows the predicted class and the selected expert. For a clean image the router chooses the identity bypass, so
nothing is changed. The classifier reaches ninety-six point eight percent accuracy on the test set. Its main
weakness is confusing clean images with lightly blurred ones, and a wrong decision sends the image to the wrong
expert.

## ch4 — Soft Mixture of Experts (~55 s)
The third workspace is the soft mixture. A gate, initialised from the classifier, gives a continuous weight to
the identity branch and the three experts, and the output is the weighted sum. The experts started from the
trained specialists and the whole system was fine-tuned jointly. Here the weight bars show salt-and-pepper
getting almost all the weight. For a lightly blurred image the gate splits its weight between the identity
branch and the blur expert, which is why the soft model is the best system on blur. No expert is inactive, and
no unrelated expert gets more than about four percent of the average weight.

## ch5 — Face-to-Sketch Generator (~65 s)
The fourth workspace is the face-to-sketch generator, a conditional GAN with a U-Net generator and a PatchGAN
discriminator. A learned style embedding feeds both networks. I can upload a photo or capture one with the
webcam, choose Style one, two or three, and generate. The photograph and the sketch appear side by side, and I
can download the result. The sketches keep the head pose and outline but are smooth and lack the hatching of the
reference drawings. This is a known limitation, and it is discussed in the report.

## ch6 — Experiments, Optuna and MLflow (~60 s)
Every task has an Optuna study, and every run is tracked in MLflow. The experiments panel shows the test
metrics and the MLflow run table. The searches were short, three to four trials of three epochs each, so the
selected configurations are the best of a small search. In the repository I have the Optuna databases, the
study summaries, the training curves, the confusion matrix, the routing heatmap, and the failure cases. The
seven ONNX models match PyTorch to within about three times ten to the minus five.

## ch7 — Results, limitations and wrap-up (~55 s)
On the official test split, restoration improves salt-and-pepper noise by about nine point six decibels and
occlusion by about eight decibels. Blur is not improved on average, because the compressed latent limits fine
detail, and the soft mixture comes closest. The sketch generator is smooth. The test split was also used to
evaluate two earlier model generations, but all selection used validation data. The source, models, Optuna and
MLflow records, tests, and the IEEE report are in the GitHub repository linked in the report. Thank you.
