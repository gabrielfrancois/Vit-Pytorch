# Vit-Pytorch: Custom Implementation of Vision Transformers and Dynamic Vision Transformers

This repository contains **our PyTorch implementation** of **Vision Transformers (ViT)** and **Dynamic Vision Transformers (DynamicViT)** for datasets such as **ImageNet** and **CIFAR-10**.


[Pruning Evolution](Vit-Pytorch/testing/log/pruning_evolution.jpeg)

DynamicViT was originally proposed in:  
*"Dynamic Vision Transformers for Efficient Image Recognition", 2021* [arXiv link](https://arxiv.org/abs/2106.02034). Our code is an independent implementation, including some modifications for adaptive pruning and efficient training on ImageNet with small teachers.

---

## Table of Contents

1. [Installation](#installation)  
2. [Project Structure](#project-structure)  
3. [How the model works](#how-the-model-works)  
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

2. Install `uv` (if it is not already done):

```bash
brew install uv
```



> Note: Using `uv` and `pyproject.toml` is cleaner than `requirements.txt`.
> You can also make a virtual environment and 
```bash
pip install -r requirements.txt
```

---

## Project Structure

* `models/`: Contains ViT and DynamicViT architectures, transformers, patch embedding, and predictor modules.
* `data/`: Data loaders for ImageNet and CIFAR-10.
* `configs/`: Training configuration files for ImageNet and CIFAR.
* `training/`: Scripts to train teacher and student models.
* `testing/`: Scripts to evaluate models and generate plots.
* `helper_function/`: Utility functions (printing, plotting, etc.).

---

## How the model works

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

---

## Usage

### Training Teacher

```bash
uv run -m training.train_teacher_imagenet
```

or : 

```bash
python3 -m training.train_teacher_imagenet
```

### Training Student (DynamicViT)

```bash
uv run -m training.train_student_imagenet
```

> Make sure to adjust `log_dir` and `checkpoint_dir` in config files to avoid overwriting previous results.

### Testing / Evaluation

```bash
uv run -m testing.test_imagenet
uv run -m testing.test_cifar
```

### Finetuning the model
```bash
uv run -m  finetuning.finetune_model
```


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
