# QIANets: Quantum-Integrated Adaptive Networks  
*Reduced Latency and Improved Inference Times in CNN Models*

[![Paper](https://img.shields.io/badge/arXiv-2410.10318-b31b1b.svg)](https://arxiv.org/pdf/2410.10318)  
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

QIANets is a **quantum-inspired model compression framework** that reduces the size and inference time of deep learning models **without sacrificing accuracy**.  
By integrating quantum-inspired pruning, tensor decomposition, and annealing-based matrix factorization, QIANets achieves highly efficient compression for convolutional neural networks (CNNs) such as **GoogLeNet, ResNet-18, and DenseNet**.

---

## 🚀 Key Features
- **Quantum-Inspired Pruning**: Removes non-essential weights using principles inspired by quantum measurement.  
- **Tensor Decomposition**: Factorizes large weight matrices into compact, efficient representations.  
- **Annealing-Based Matrix Factorization**: Uses a quantum-inspired annealing process to optimize compression while preserving accuracy.  
- **Low Latency & High Efficiency**: Achieves significant reductions in inference time with minimal accuracy loss.  

---

## 📑 Table of Contents
- [Overview](#qianets-quantum-integrated-adaptive-networks)  
- [Key Features](#-key-features)  
- [Getting Started](#-getting-started)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Quantum-Inspired Techniques](#-quantum-inspired-techniques)  
- [Results](#-results)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Contact](#-contact)  

---

## 🛠 Getting Started

### Prerequisites
- Python 3.x  
- TensorFlow or PyTorch  
- *(Optional but helpful)* Qiskit or similar quantum computing libraries  


## 💻 Installation

1. Clone the repository
```
git clone https://github.com/vanessamatvei/QIANets
cd QIANets
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Set up your environment for quantum-inspired computations
[Optional] Install quantum computing libraries such as Qiskit for deeper exploration of the quantum principles.

---

### ▶️ Usage
1. Train a base CNN model (e.g., ResNet-18) using the provided dataset:
```
python train.py --dataset <dataset> --model <model-type>
```

2. Compress the model using quantum-inspired techniques:
```
python compress.py --model <trained-model-path> --compression-rate <rate>
```

3. Evaluate the compressed model:
```
python evaluate.py --model <compressed-model-path> --dataset <dataset>
```

---

### Example Workflow
1. Train a model:
```
python train.py --dataset cifar10 --model resnet18
```

2. Compress using a 75% rate:
```
python compress.py --model models/resnet18.h5 --compression-rate 0.75
```

3. Evaluate:
```
python evaluate.py --model models/resnet18_compressed.h5 --dataset cifar10
```

---

### 🧩 Quantum-Inspired Techniques

1. **Quantum-Inspired Pruning**  
   We use ideas from quantum measurement and probabilistic importance scoring to identify and remove weights that contribute little to a model’s predictive fidelity. This reduces parameter count and compute without materially degrading accuracy.

2. **Tensor Decomposition**  
   Large weight tensors are factorized into smaller component matrices/tensors (e.g., CP / Tucker / SVD-style factorizations). This produces compact representations that keep the most informative directions while lowering memory and compute costs.

3. **Annealing-Based Matrix Factorization**  
   A simulated-annealing style optimizer (inspired by quantum annealing) searches the factorization space to find low-error compressions. This helps balance aggressive compression with accuracy preservation by escaping poor local minima.

---

### 📊 Results (summary)

Tested on GoogLeNet, ResNet-18, and DenseNet, QIANets demonstrated:

- **50–70% reduction in inference time** (measured on target hardware)  
- **Up to ~80% compression** with minimal accuracy loss when tuned appropriately  
- Faster, more reliable deployment on resource-constrained devices (mobile / edge)  
- Strong promise for real-time and low-latency AI applications

---

### 🤝 Contributing
We welcome contributions! To get involved:

1. Fork the repo and create a new branch:
```
git checkout -b feature/your-feature
```

2. Commit your changes:
```
git add .
git commit -m "Add your feature"
```

3. Push your branch:
```
git push origin feature/your-feature
```

4. Open a Pull Request for review.
Please follow the repository's CONTRIBUTING.md (if present) and write tests where applicable.

---

### 📜 License
This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

---

### 📬 Contact
For questions or collaboration opportunities, reach out to:

Vanessa Matvei

Email: vanessamatvei@gmail.com 


If you use QIANets in academic work, please cite the paper: arXiv:2410.10318.

---

Thank you for your interest in QIANets!
