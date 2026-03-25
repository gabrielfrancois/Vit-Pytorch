# Vit-Pytorch: Custom Implementation of Dynamic Vision Transformers

![Pruning Evolution](logs/imagenet/student/pruning/pruning_evolution.jpeg)

**Dynamic Token Pruning: Overcoming the computational cost of Transformers to enable high-performance inference on CPU-limited devices by dynamically selecting only the most informative image tokens.**

This repository contains a PyTorch implementation of Vision Transformers (ViT) and Dynamic Vision Transformers (DynamicViT) for datasets such as ImageNet-1K, STL-10, and CIFAR-10.
For more details on the theory and implementation, please refer to the project report "**Projet_ViT.pdf**".

DynamicViT was originally proposed in:  
*"Dynamic Vision Transformers for Efficient Image Recognition", 2021* [arXiv link](https://arxiv.org/abs/2106.02034). This repository is an independent implementation featuring adaptive pruning schedules and knowledge distillation strategies.

## Table of Contents

1. [Installation](#installation)  
2. [Project Structure](#project-structure)  
3. [How the model works](#how-the-model-works)  
4. [Usage](#usage)  
5. [Citation](#citation)

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd Vit-Pytorch
```

2. Set up a virtual environment and install dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (MacOS/Linux)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Alternatively, if you use `uv`:
```bash
uv sync
```

## Project Structure

```text
Vit-Pytorch/
├── src/
│   ├── models/           # Architecture: ViT, DynamicViT, Patch Embedding, Predictor LG
│   ├── train/            # Training loops for Teacher and Student (CIFAR & ImageNet)
│   ├── test/             # Evaluation scripts and parameter computing
│   └── finetuning/       # LoRA fine-tuning scripts for STL-10
├── configs/              # Hyperparameters and data paths per dataset
├── data/                 # Data loading logic and raw data storage
├── logs/                 # TensorBoard events and generated plots
├── checkpoints/          # Saved model weights (.pth)
├── helper_function/      # Utility functions for printing and formatting
├── pyproject.toml        # Project dependencies and metadata
└── requirements.txt      # Traditional pip requirements
```

## How the model works

### Dynamic Vision Transformer (DynamicViT)
Dynamic ViT extends the standard ViT with dynamic token pruning to reduce FLOPs and memory usage. Since not all image patches contribute equally to classification, the model learns to prune less informative tokens in deeper layers.

**Adaptive pruning ratio (rho)**: To stabilize training, we implement a sigmoid-based pruning schedule. In early epochs, the model retains most tokens to learn robust features. The pruning intensity gradually increases towards a target ratio (`rho_final`) in the middle of training before stabilizing in the final epochs.

### Loss Function
The training objective combines four components:
1. **Classification Loss**: Standard cross-entropy for the target task.
2. **Knowledge Distillation**: Matching the student's features to a frozen teacher model.
3. **KL Divergence**: Aligning the probability distributions of the student and teacher.
4. **Ratio Loss**: Enforcing the target token keep-ratio at each pruning stage.

## Usage

All scripts must be executed from the project root using the `-m` flag to ensure proper module resolution.

### 1. Training a Teacher Model
Train a standard Vision Transformer to serve as a teacher:
```bash
python -m src.train.train_teacher_cifar
# or for ImageNet
python -m src.train.train_teacher_imagenet
```

### 2. Training a Student Model (DynamicViT)
Train a student model with dynamic pruning, distilling knowledge from a pretrained teacher:
```bash
python -m src.train.train_student_cifar
# or for ImageNet
python -m src.train.train_student_imagenet
```

### 3. Evaluation
Evaluate model performance, compute FLOPs/Parameters, and generate confusion matrices:
```bash
python -m src.test.test_cifar
python -m src.test.test_imagenet
```

### 4. Fine-tuning
Fine-tune a pretrained model on a new dataset (e.g., STL-10) using LoRA:
```bash
python -m src.finetuning.finetune_model
```

## Citation

If you use this implementation, please cite this repository and the original DynamicViT paper:

```text
@article{DBLP:journals/corr/abs-2106-02034,
  author       = {Yongming Rao and
                  Wenliang Zhao and
                  Benlin Liu and
                  Jiwen Lu and
                  Jie Zhou and
                  Cho{-}Jui Hsieh},
  title        = {DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification},
  journal      = {CoRR},
  volume       = {abs/2106.02034},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.02034},
  eprinttype    = {arXiv},
  eprint       = {2106.02034},
  timestamp    = {Thu, 10 Jun 2021 16:34:18 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2106-02034.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
