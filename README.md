# Marketing_analytics_-pytorch_and_-CNN-

# PyTorch CNNs for Fashion-MNIST

A progressive deep learning project exploring neural network architectures for image classification on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset — from a simple MLP baseline all the way to ResNet18 transfer learning.

---

## Dataset

**Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories:

`T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot`

| Split | Samples |
|-------|---------|
| Train | 60,000 |
| Test | 10,000 |

- Normalised: mean = 0.2859, std = 0.3530
- Batch size: 64
- Optimizer: Adam (lr = 0.001)
- Loss: CrossEntropyLoss

---

## Project Structure

### 1. MLP Baseline
A fully connected multi-layer perceptron as a starting point.

```
Flatten → Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)
```
- ~235K parameters
- ~88% test accuracy

---

### 2. LeNet-5
Classic 1998 CNN architecture adapted for Fashion-MNIST.

```
Conv(1→6, 5×5) + Sigmoid → AvgPool
Conv(6→16, 5×5) + Sigmoid → AvgPool
Linear(400→120) → Linear(120→84) → Linear(84→10)
```
- ~61K parameters
- ~88–89% test accuracy

---

### 3. Modernised LeNet
LeNet-5 updated with modern deep learning practices.

**Changes:**
- Sigmoid activations → **ReLU**
- Average pooling → **Max pooling**

Same architecture, meaningfully better convergence and accuracy.

---

### 4. Custom CNN
A purpose-built CNN targeting 90%+ accuracy, designed from scratch.

```
Conv(1→32, 3×3) + BatchNorm + ReLU
Conv(32→64, 3×3) + BatchNorm + ReLU → MaxPool(2×2)
Conv(64→128, 3×3) + BatchNorm + ReLU
Conv(128→128, 3×3) + BatchNorm + ReLU → MaxPool(2×2)
Flatten → Linear(6272→512) → ReLU → Dropout(0.5) → Linear(512→10)
```

**Design decisions:**
- 4 convolutional blocks with 3×3 kernels (more expressive than 5×5 at lower cost)
- Batch normalisation after every conv layer (stabilises training, allows higher lr)
- Pooling only twice to preserve spatial information longer
- Dropout(0.5) in the classifier to control overfitting

- ~3.3M parameters
- ~93–94% test accuracy

---

### 5. Transfer Learning — ResNet18
Three experiments using ResNet18 to explore transfer learning strategies.

| Approach | Weights | Trainable Layers | Notes |
|----------|---------|-----------------|-------|
| Random init | None | All | Slower convergence, lower accuracy |
| Full fine-tune | ImageNet pretrained | All | Fastest convergence, highest accuracy |
| Frozen backbone | ImageNet pretrained | FC layer only | Least overfitting, slightly lower accuracy |

Input resized to 224×224, converted to 3 channels, ImageNet normalisation applied.

**Key takeaway:** Pretrained weights provide a strong boost over random initialisation. Freezing the backbone reduces overfitting risk at a small accuracy cost — useful when compute or data is limited.

---

## Results Summary

| Model | Parameters | ~Test Accuracy |
|-------|-----------|---------------|
| MLP | ~235K | ~88% |
| LeNet-5 | ~61K | ~88–89% |
| Modern LeNet | ~61K | ~89–90% |
| Custom CNN | ~3.3M | ~93–94% |
| ResNet18 (random) | ~11M | ~85–88% |
| ResNet18 (pretrained) | ~11M | ~90–92% |
| ResNet18 (frozen) | ~11M (few trained) | ~88–90% |

---

## Requirements

```bash
pip install torch torchvision torchsummary matplotlib
```

> Recommended: Run on **Google Colab** with a GPU runtime for reasonable training times.
> `Runtime → Change Runtime Type → GPU`

---

## Running the Notebook

1. Open `CNN_Pytorch_for_Fashion_analytics.ipynb` in Jupyter or Google Colab
2. Run cells top to bottom — data downloads automatically via `torchvision`
3. Each section is self-contained and trains independently
