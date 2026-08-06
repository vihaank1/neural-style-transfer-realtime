# Real-Time Neural Style Transfer (PyTorch + OpenCV)

A feed-forward neural style transfer system trained from scratch, deployed in a
real-time OpenCV video pipeline. No pretrained backbone (e.g. VGG) is used —
the perceptual feature space used for training is itself a small convolutional
autoencoder trained from scratch on real photographs.

## Why no pretrained VGG?

Classic fast neural style transfer (Johnson et al., 2016) computes content and
style loss using features from an ImageNet-pretrained VGG16. This project
intentionally avoids depending on any externally hosted pretrained weights and
instead learns its own feature space end-to-end:

1. **Self-supervised feature encoder** — a small conv autoencoder
   (`src/autoencoder.py`) is trained via image reconstruction (MSE) on ~600
   diverse real-world photos. Its encoder is then frozen and used as the
   "perceptual" feature extractor, in place of VGG.
2. **Feed-forward style transfer network** — a residual CNN
   (`src/transformer_net.py`, architecture after Johnson et al.) is trained to
   map a content image to a stylized image in a single forward pass, using
   content loss + Gram-matrix style loss + total-variation loss computed from
   the self-trained encoder's activations.
3. **Real-time inference pipeline** — `src/realtime_pipeline.py` runs the
   trained network over video frames captured with OpenCV, batching frames and
   using `torch.inference_mode()` for throughput, with FPS benchmarking.

## Results

Trained on a CPU-only, 4-core / 4GB sandbox (no GPU) — everything below reflects
that hardware, not a production training rig.

**Autoencoder** (15 epochs, 600 images, 128x128): reconstruction MSE loss
0.049 → 0.020.

**Style network** (10 epochs, 600 content images, style = `mosaic.jpg`):
content loss 0.224 → 0.119, converging steadily (see
`checkpoints/style_mosaic_log.json`).

**Real-time pipeline benchmark** (`checkpoints/pipeline_benchmark.json`,
795-frame test video, 4 CPU threads, batch size 8):

| Metric | Value |
|---|---|
| Naive single-frame inference (grad tracking on) | 55.0 FPS |
| Batched + `inference_mode()` | 64.7 FPS |
| Source video frame rate | 10 FPS |
| **Real-time multiplier** | **6.5x** |

See `samples/final_comparison.jpg` for content/stylized pairs on held-out
images never seen during training, and `samples/stylized_vtest_full.mp4` for
the full stylized video output.

## Honest limitations

This is a portfolio-scale project, not a production system: 128x128
resolution, a few hundred training images, and ~10 epochs on CPU. Output
quality is recognizably stylized but not as crisp as the original paper's
results (trained on ~80K COCO images with a GPU for many more iterations).
The architecture and training loop are correct and would scale directly to
more data / more compute / a GPU.

## Project structure

```
src/
  autoencoder.py         # self-supervised feature encoder + decoder
  train_autoencoder.py   # stage 1: train the encoder via reconstruction
  transformer_net.py     # feed-forward style transfer network
  losses.py               # gram matrix + total variation
  train_style.py          # stage 2: train the style network
  infer.py                # single-image inference + before/after grids
  make_comparison.py      # build the final comparison image
  realtime_pipeline.py    # stage 3: OpenCV video pipeline + FPS benchmark
checkpoints/               # trained weights + training logs (real, not mocked)
samples/                   # before/after images + stylized demo video
data/style/mosaic.jpg      # style target (from pytorch/examples, BSD-licensed)
```

## Reproducing

```
pip install -r requirements.txt

# get training images (~1 per ImageNet class, used as content images)
git clone https://github.com/EliSchwartz/imagenet-sample-images.git

# stage 1: train the feature encoder
python src/train_autoencoder.py --list <file_list.txt> --epochs 15

# stage 2: train the style network
python src/train_style.py --list <file_list.txt> --style data/style/mosaic.jpg --epochs 10

# stage 3: run the real-time video pipeline + benchmark
python src/realtime_pipeline.py --video data/vtest.avi --ckpt checkpoints/style_mosaic.pt
```

## Attribution

- Style image (`mosaic.jpg`) and demo video source pattern from
  [pytorch/examples](https://github.com/pytorch/examples) (`fast_neural_style`, BSD license).
- Test video `vtest.avi` from the [OpenCV](https://github.com/opencv/opencv) sample data (Apache 2.0).
- Content training images from
  [EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images).
