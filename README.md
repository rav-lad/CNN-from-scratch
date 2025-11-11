# CNN from Scratch in NumPy (No Autograd)

**Goal.** Build and train a small convolutional neural network **from scratch** in NumPy.
No autograd. Every forward and backward pass is implemented by hand, including convolution, pooling, batch normalization, and softmax cross entropy.

## 🆕 DermaScan - AI Skin Condition Diagnosis

**NEW!** This repository now includes **DermaScan**, a web application that uses the CNN infrastructure to detect skin conditions from images.

👉 **[See DermaScan Documentation](dermascan/README.md)** | **[View Plan](DERMASCAN_PLAN.md)**

Quick start:
```bash
pip install -r dermascan/requirements.txt
bash dermascan/scripts/run_server.sh
# Open http://localhost:8000
```

## Why this project matters

- Proves **mathematical understanding** of gradients and optimization.
- Demonstrates **clean software design** and testable components.
- Shows **industry readiness** with a reproducible training pipeline and CLI.

## Features

- Layers: `Conv2D`, `MaxPool2D`, `Dense`, `BatchNorm2D`, `Dropout`, common activations.
- Core: manual losses, optimizers (SGD, Momentum, Adam), schedulers, callbacks.
- Utilities: `im2col` and `col2im`, one hot, batching, seed control.
- Datasets: MNIST and CIFAR-10 loaders.
- CLI: `train.py`, `evaluate.py`, `export.py`.
- Tests: strict unit tests with numerical gradient checks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

## Quick start

Train a small LeNet on MNIST:

```bash
python -m src.cli.train --config src/configs/mnist_lenet.yaml
```

Evaluate on the test set:

```bash
python -m src.cli.evaluate --config src/configs/mnist_lenet.yaml --weights checkpoints/lenet_mnist_best.npz
```

Export weights and architecture:

```bash
python -m src.cli.export --config src/configs/mnist_lenet.yaml --weights checkpoints/lenet_mnist_best.npz
```

## Project structure

```
cnn-from-scratch/
├─ pyproject.toml
├─ README.md
├─ DERMASCAN_PLAN.md        # 🆕 DermaScan project plan
├─ data/                    # raw/ and processed/ datasets
│  └─ dermatology/          # 🆕 HAM10000 dataset
├─ dermascan/               # 🆕 Skin diagnosis application
│  ├─ api/                  # FastAPI server
│  ├─ inference/            # Model predictions
│  ├─ preprocessing/        # Image processing
│  ├─ database/             # Medical conditions info
│  ├─ configs/              # Model configurations
│  └─ scripts/              # Utilities
├─ frontend/                # 🆕 Web interface
│  ├─ static/               # CSS, JS
│  └─ templates/            # HTML
├─ notebooks/               # sanity checks, training report
├─ src/
│  ├─ core/                 # tensor utils, inits, losses, optim, metrics
│  ├─ layers/               # conv, pool, dense, activations, bn, dropout
│  ├─ models/               # sequential, small presets
│  ├─ data/                 # mnist, cifar10
│  └─ train/                # loop, callbacks, scheduler, logger, cli
├─ tests/                   # unit and numeric gradient checks
└─ reports/                 # figures and CSV logs
```

## Math overview

* **Convolution** via `im2col` trick to reduce conv to GEMM.
  Shapes: input `(N, C_in, H, W)`, weights `(C_out, C_in, KH, KW)`, output `(N, C_out, H_out, W_out)`.
* **Backward Conv**: compute `dW` from cached columns and `dX` via `col2im`.
* **Pooling**: max pool with index mask for backward, average pool with uniform gradient.
* **Softmax Cross Entropy**: stable softmax with logits shift. Derive analytic gradient `dL/dlogits = p - y_onehot`.
* **BatchNorm2D**: per channel mean and variance, running stats in eval mode, full backward.

References are cited in code comments and notebooks.

## Reproducibility

* Seed all RNGs and log them.
* Save configs, metrics, and checkpoints.
* Use `reports/results.csv` for runs. Figures go to `reports/figures/`.

## Testing

Run all tests:

```bash
pytest -q
```

Key tests check:

* `im2col` and `col2im` inverses on valid windows.
* `Conv2D.backward` vs numerical gradients.
* Pooling backward correctness.
* BatchNorm backward against finite differences.
* Loss gradients for MSE and softmax CE.

## Configs

Example `src/configs/mnist_lenet.yaml`:

```yaml
dataset: mnist
model: lenet_mnist
seed: 42
train:
  epochs: 15
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 0.0005
  scheduler:
    name: step
    step_size: 5
    gamma: 0.5
callbacks:
  - early_stopping:
      monitor: val_acc
      patience: 3
  - checkpoint:
      filepath: "checkpoints/lenet_mnist_best.npz"
      monitor: val_acc
      mode: max
```

## Milestones

### Core CNN (Completed)
* [✅] `im2col` and `col2im` with tests
* [✅] `Conv2D` forward and backward
* [✅] `MaxPool2D` forward and backward
* [✅] `Dense`, `ReLU`, `SoftmaxCE` with gradients
* [✅] Train LeNet on MNIST to >98% val acc
* [✅] BatchNorm2D + Dropout
* [✅] Tiny VGG on CIFAR-10 to a reasonable baseline

### DermaScan (New Project)
* [✅] Project structure and architecture
* [✅] FastAPI backend with REST endpoints
* [✅] Image preprocessing pipeline
* [✅] CNN model architecture for dermatology
* [✅] Medical conditions database (7 classes)
* [✅] Web frontend (HTML/CSS/JS)
* [✅] Complete documentation
* [⏳] HAM10000 data loader
* [⏳] Model training pipeline
* [⏳] Production deployment

## Limitations

* Educational focus, not optimized for speed.
* No mixed precision, no GPU, no autograd.
* Only small models and datasets.

## License

MIT

