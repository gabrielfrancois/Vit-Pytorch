

# Vit-Pytorch: Custom Implementation of Vision Transformers and Dynamic Vision Transformers

This repository contains **our PyTorch implementation** of **Vision Transformers (ViT)** and **Dynamic Vision Transformers (DynamicViT)** for datasets such as **ImageNet** and **CIFAR-10**.


![Pruning Evolution](/testing/log/imagenet1k/Pruning_Images_ImageNet/img_0.png)

DynamicViT was originally proposed in:  
*"Dynamic Vision Transformers for Efficient Image Recognition", 2021* [arXiv link](https://arxiv.org/abs/2106.02034). Our code is an independent implementation, including some modifications for adaptive pruning and efficient training on ImageNet with small teachers.

---

## Table of Contents

1. [Installation](#installation)  
2. [Project Structure](#project-structure)  
3. [How It Works](#how-it-works)  
4. [Usage](#usage)  
5. [Training](#training)  
6. [Evaluation](#evaluation)  
7. [Visualization](#visualization)  
8. [Citation](#citation)

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd Vit-Pytorch
````

2. Install dependencies using `uv` (optional):

```bash
pip install uv
uv pip install -r pyproject.toml
```

3. Activate virtual environment:

```bash
source .venv/bin/activate
```

> Note: Using `uv` and `pyproject.toml` is cleaner than `requirements.txt`.

---

## Project Structure

* `models/`: Contains ViT and DynamicViT architectures, transformers, patch embedding, and predictor modules.
* `data/`: Data loaders for ImageNet and CIFAR-10.
* `configs/`: Training configuration files for ImageNet and CIFAR.
* `training/`: Scripts to train teacher and student models.
* `testing/`: Scripts to evaluate models and generate plots.
* `helper_function/`: Utility functions (printing, plotting, etc.).

---

## How It Works

### Vision Transformer (ViT)

* Splits input images into patches and embeds them (`d_model`).
* Adds **positional encoding**.
* Processes embeddings with a **stack of transformer encoders**.
* The **CLS token** summarizes the image for classification.
* Outputs logits via a linear classification head.
* Trained with standard **cross-entropy loss**.

### Dynamic Vision Transformer (DynamicViT)

* Extends ViT with **dynamic token pruning** to reduce FLOPs and memory usage.
* Each layer predicts which tokens can be pruned in deeper layers.
Voici une version réécrite et plus fluide :

* **Adaptive pruning ratio (`rho`)**:

  * Adjusted according to a **sigmoid schedule** during training.
  * **Early epochs**: only a few tokens are pruned, allowing the model to learn robust features.
  * **Middle epochs**: more tokens are gradually pruned to improve efficiency.
  * **Late epochs**: pruning slows down to stabilize the model.
  * This strategy makes it possible to train DynamicViT on ImageNet even when using a **small teacher model**.

* Loss function combines:

  1. Classification loss
  2. Knowledge distillation from teacher
  3. KL divergence
  4. Ratio loss (enforces target keep ratio per layer)
* Only unpruned tokens are passed through deeper layers; CLS token is always kept.

---

## Usage

### Training Teacher

```bash
python -m training.train_teacher_imagenet
```

### Training Student (DynamicViT)

```bash
python -m training.train_student_imagenet
```

> Make sure to adjust `log_dir` and `checkpoint_dir` in config files to avoid overwriting previous results.

### Testing / Evaluation

```bash
python -m testing.test_imagenet
python -m testing.test_cifar
```

### Running Unit Tests

```bash
python -m pytest tests/ -v
python -m pytest -v tests/test_Vit/test_attention_head.py
```

---

## Visualization

* Training & validation loss and accuracy curves.
* Confusion matrices for teacher and student.
* Per-class accuracy comparison.
* Visualization of pruning masks on input images.

---

## Citation

If you use this implementation, please cite our repository and refer to the original DynamicViT paper:

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




