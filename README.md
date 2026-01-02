# HIFN: Hybrid Information Flow Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TNNLS-green.svg)](paper/HIFN_general_paper.pdf)

> **A Novel Deep Learning Framework with Jointly Learnable Information-Theoretic Parameters for Interpretable Classification**

<p align="center">
  <img src="assets/hifn_architecture.png" alt="HIFN Architecture" width="800"/>
</p>

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

## 🎯 Overview

**HIFN (Hybrid Information Flow Networks)** is a novel deep learning framework that extends the Variational Information Bottleneck (VIB) by introducing **four jointly learnable information-theoretic parameters per layer**:

| Parameter | Symbol | Description | Innovation |
|-----------|--------|-------------|------------|
| **Information Retention** | β | Controls compression-prediction trade-off | Learnable (vs. fixed in VIB) |
| **Entropy Budget** | B | Maximum information capacity per layer | Novel constraint |
| **Compression Ratio** | c | Adaptive dimensionality reduction [0.3, 1.0] | Novel parameter |
| **Information Gates** | g | Per-neuron importance weights | Static learnable (vs. input-dependent in LSTM) |

Unlike standard VIB where β is a fixed hyperparameter requiring manual tuning, **all HIFN parameters are optimized end-to-end** via backpropagation.

## 🚀 Key Innovation

### HIFN vs. Standard VIB

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Standard VIB                                        │
│  ┌─────────┐      ┌─────────────┐      ┌─────────┐                          │
│  │  Input  │ ──── │   Encoder   │ ──── │    z    │     β = FIXED (manual)   │
│  │    x    │      │   p(z|x)    │      │         │     No gates             │
│  └─────────┘      └─────────────┘      └─────────┘     No budget            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  Our Extension
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HIFN (This Work)                                    │
│  ┌─────────┐      ┌─────────────┐      ┌─────────┐                          │
│  │  Input  │ ──── │  IB Layer   │ ──── │    z    │     β = LEARNABLE        │
│  │    x    │      │ β,B,c,g     │      │         │     g = INFO GATES       │
│  └─────────┘      └─────────────┘      └─────────┘     B = ENTROPY BUDGET   │
│                                                        c = COMPRESSION      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Information Gates are Different from LSTM Gates

| Property | LSTM Gates | HIFN Gates |
|----------|------------|------------|
| **Computation** | `g = σ(W·[h,x] + b)` | `g = learnable parameter` |
| **Type** | Input-dependent function | Static parameter |
| **Varies per sample** | ✅ Yes | ❌ No |
| **Interpretability** | Local (per sample) | **Global** (across all samples) |
| **Purpose** | Memory control | **Feature importance** |

## 🏗️ Architecture

### Information Bottleneck Layer

```
                    ┌─────────────────────────────────────────────────────────┐
                    │            Information Bottleneck Layer                  │
                    │                                                          │
    ┌───┐           │  ┌────┐    ┌────────┐    ┌───┐    ┌─────┐    ┌────┐    │    ┌───┐
    │ x │ ────────► │  │ Wμ │───►│   μ    │───►│ ⊙ │───►│  +  │───►│×c_ℓ│───►│───►│ z │
    └───┘           │  └────┘    └────────┘    └─┬─┘    └──┬──┘    └────┘    │    └───┘
                    │                            │         │                  │
                    │  ┌────┐    ┌────────┐      │      ┌──┴──┐               │
                    │  │ Wσ │───►│ log σ² │──────┼─────►│  ε  │               │
                    │  └────┘    └────────┘      │      └─────┘               │
                    │                            │      N(0,I)                │
                    │                     ┌──────┴──────┐                     │
                    │                     │   σ(g_ℓ)    │ ◄── INFO GATES      │
                    │                     │  (Novel)    │     (Learnable)     │
                    │                     └─────────────┘                     │
                    └─────────────────────────────────────────────────────────┘
                    
    Novel Parameters: β_ℓ (in loss), B_ℓ (in loss), c_ℓ (compression), g_ℓ (gates)
```

### Complete HIFN Network

```python
HIFN(
    input_dim=784,           # Input features
    hidden_dims=[256, 128, 64],  # IB layer dimensions
    num_classes=10,          # Output classes
    beta_init=0.01           # Initial β value
)
```

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- scikit-learn

### Install via pip

```bash
# Clone the repository
git clone https://github.com/mtawfik/HIFN.git
cd HIFN

# Install dependencies
pip install -r requirements.txt

# Install HIFN package
pip install -e .
```

### Requirements file

```txt
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.62.0
```

## ⚡ Quick Start

```python
import torch
from hifn import HIFN, HIFNTrainer

# Create model
model = HIFN(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    num_classes=10,
    beta_init=0.01
)

# Create trainer
trainer = HIFNTrainer(
    model=model,
    learning_rate=0.001,
    beta_weight=0.0001,
    device='cuda'
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)

# Get predictions
predictions, targets = trainer.predict(test_loader)

# Get feature importance from learned gates
importance = model.get_feature_importance()
```

## 📖 Usage Examples

### Example 1: MNIST Classification

```python
from hifn import HIFN, HIFNTrainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# Build HIFN
model = HIFN(
    input_dim=784,
    hidden_dims=[256, 128, 64],
    num_classes=10
)

# Train
trainer = HIFNTrainer(model, learning_rate=0.001, device='cuda')
history = trainer.fit(train_loader, test_loader, epochs=50)

# Evaluate
val_loss, val_acc = trainer.validate(test_loader)
print(f"Test Accuracy: {val_acc:.2f}%")
```

### Example 2: Network Intrusion Detection (CICIDS-2017)

```python
from hifn import HIFN, HIFNTrainer, HIFNVisualizer
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess CICIDS-2017
df = pd.read_csv('cicids2017.csv')
X = df.drop('Label', axis=1).values
y = LabelEncoder().fit_transform(df['Label'])

# Scale features
X = StandardScaler().fit_transform(X)

# Create data loaders
# ... (split and create DataLoader)

# Build HIFN for intrusion detection
model = HIFN(
    input_dim=78,  # CICIDS features
    hidden_dims=[256, 128, 64],
    num_classes=15,  # Attack types
    beta_init=0.01
)

# Train and visualize
trainer = HIFNTrainer(model, device='cuda')
history = trainer.fit(train_loader, test_loader, epochs=30)

# Visualize information flow
HIFNVisualizer.plot_information_flow(model, sample_batch)

# Get feature importance
fig, indices, scores = HIFNVisualizer.plot_feature_importance(
    model, feature_names, top_k=20
)
```

### Example 3: Extracting Feature Importance

```python
# After training, get interpretability from gates
model.eval()

# Get gate values from first layer
first_layer = model.layers[0]
gate_importance = torch.sigmoid(first_layer.info_gates_logit).detach().cpu().numpy()

# Map to input features
input_weights = first_layer.encoder_mean.weight.detach().cpu().numpy()
feature_scores = np.abs(input_weights).T @ gate_importance

# Rank features
top_features = np.argsort(feature_scores)[::-1][:20]
print("Top 20 most important features:", top_features)
```

## 🧪 Experiments

### Datasets

| Dataset | Domain | Samples | Features | Classes |
|---------|--------|---------|----------|---------|
| MNIST | Image | 70,000 | 784 | 10 |
| CIFAR-10 | Image | 60,000 | 3,072 | 10 |
| CICIDS-2017 | Network Security | 2.8M | 78 | 15 |

### Running Experiments

```bash
# MNIST experiment
python examples/mnist_example.py

# CICIDS-2017 experiment
python examples/cicids2017_example.py

# Run all experiments
python experiments/run_all.py
```

## 📊 Results

### Classification Accuracy

| Method | MNIST | CIFAR-10 | CICIDS-2017 |
|--------|-------|----------|-------------|
| Standard MLP | 98.21% | 52.34% | 97.82% |
| Dropout MLP | 98.45% | 53.12% | 98.15% |
| VIB (β=0.001) | 98.34% | 52.87% | 97.95% |
| VIB (β=0.01) | 98.12% | 51.23% | 97.45% |
| **HIFN (Ours)** | **98.67%** | **54.21%** | **98.45%** |

### Learned Parameters (CICIDS-2017)

| Layer | β (retention) | B (budget) | c (compression) | H (entropy) |
|-------|---------------|------------|-----------------|-------------|
| IB Layer 1 | 0.0123 | 512.3 bits | 0.89 | 38.7 bits |
| IB Layer 2 | 0.0156 | 256.1 bits | 0.67 | 24.3 bits |
| IB Layer 3 | 0.0189 | 128.4 bits | 0.43 | 14.8 bits |

**Key Observation**: β increases with depth (deeper layers retain more), c decreases (deeper layers compress more) — patterns emerged automatically through learning!

### Ablation Study

| Configuration | Accuracy | Δ |
|--------------|----------|---|
| Full HIFN | **98.45%** | — |
| w/o Info Gates g | 97.34% | -1.11% |
| w/o Compression c | 97.80% | -0.65% |
| w/o Learnable β | 97.91% | -0.54% |
| w/o Entropy Budget B | 98.02% | -0.43% |
| w/o All Novel (= VIB) | 97.12% | -1.33% |

## 📂 Project Structure

```
HIFN/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── hifn/
│   ├── __init__.py
│   ├── core.py              # HIFN model implementation
│   ├── layers.py            # InformationBottleneckLayer
│   ├── trainer.py           # HIFNTrainer class
│   └── visualize.py         # HIFNVisualizer utilities
├── examples/
│   ├── mnist_example.py
│   ├── cifar10_example.py
│   └── cicids2017_example.py
├── experiments/
│   ├── run_all.py
│   └── ablation_study.py
├── paper/
│   ├── HIFN_general_paper.pdf
│   └── HIFN_general_paper.tex
└── assets/
    ├── hifn_architecture.png
    └── results/
```

## 📄 Citation

If you use HIFN in your research, please cite:

```bibtex
@article{tawfik2025hifn,
  title={HIFN: Hybrid Information Flow Networks with Jointly Learnable 
         Information-Theoretic Parameters for Interpretable Deep Learning},
  author={Tawfik, Mohammed},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}
```

## 👤 Author

**Mohammed Tawfik**

- 📧 Email: [kmkhol01@gmail.com](mailto:kmkhol01@gmail.com)
- 🏫 Affiliation: Department of Cybersecurity and Cloud Computing, Ajloun National University, Jordan
- 🎓 Member, IEEE

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 
- The authors of VIB [Alemi et al., 2016] for foundational work
- PyTorch team for the excellent deep learning framework

---

<p align="center">
  <b>⭐ Star this repository if you find HIFN useful! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by Mohammed Tawfik
</p>
