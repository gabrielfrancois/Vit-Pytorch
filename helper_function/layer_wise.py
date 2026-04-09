from torch import nn as nn
from collections import defaultdict

from helper_function.print import *


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_id(name: str) -> int:
    """
    Map a parameter name to its layer index.
    Transformer encoder layers are indexed from 1.
    Everything else (embeddings, head, etc.) gets index 0.
    Args:
        name (str): Full parameter name from model.named_parameters(),
                    e.g. 'transformer_encoder.3.attention.qkv.weight'.
    Returns:
        layer_id (int): Layer index >= 0.
    """
    if "transformer_encoder." in name:
        return int(name.split("transformer_encoder.")[1].split(".")[0]) + 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise LR Schedules
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_increasing(model: nn.Module, alpha: float) -> list[dict]:
    """
    Assign learning rates by layer block using a polynomial (power) schedule.
    Layers 1–3 get alpha, layers 4–7 get alpha², deeper layers get alpha³.
    This gives strong LR to early layers and progressively lower LR to deeper ones.
    Args:
        model (nn.Module): The model whose parameters are being grouped.
        alpha (float):     Base learning rate. Must be in (0, 1) so powers decay.
    Returns:
        param_groups (list[dict]): List of dicts with keys 'params', 'lr', 'weight_decay'.
                                   Ready to pass directly to any torch.optim optimizer.
    """
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = get_layer_id(name)

        if layer_id <= 3:
            lr = alpha
        elif layer_id <= 7:
            lr = alpha ** 2
        else:
            lr = alpha ** 3
        param_groups.append({"params": [param], "lr": lr, "weight_decay": 1e-4})
    print(blue("LAYER-WISE:"))
    for layer, g in enumerate(param_groups):
        print(f'learning rate for layer {layer+1}: {g["lr"]}')
    return param_groups


def increasing_llrd(model: nn.Module, alpha: float, layer_decay: float, num_layers: int) -> list[dict]:
    """
    Layer-wise LR decay where early layers get the lowest LR and later layers get higher LR.
    LR at layer i = alpha * (layer_decay ^ (num_layers - i)).
    Parameters sharing the same computed LR are grouped into one param group.
    Args:
        model       (nn.Module): The model whose parameters are being grouped.
        alpha       (float):     Base (maximum) learning rate applied to the last layer.
        layer_decay (float):     Multiplicative decay factor per layer, typically in (0, 1).
        num_layers  (int):       Total number of transformer layers. Layer IDs are clamped to this.
    Returns:
        param_groups (list[dict]): List of dicts with keys 'params', 'lr', 'weight_decay'.
                                   Ready to pass directly to any torch.optim optimizer.
    """
    group_dict = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = min(get_layer_id(name), num_layers)
        lr = alpha * (layer_decay ** (num_layers - layer_id))
        group_dict[lr].append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": 1e-4}
        for lr, params in group_dict.items()
    ]
    print(blue("LAYER-WISE:"))
    for layer, g in enumerate(param_groups):
        print(f'learning rate for layer {layer+1}: {g["lr"]}')
    return param_groups


def decreasing_llrd(model: nn.Module, alpha: float, layer_decay: float, num_layers: int) -> list[dict]:
    """
    Standard LLRD: early layers get the highest LR and later layers get lower LR.
    LR at layer i = alpha * (layer_decay ^ i).
    Parameters sharing the same computed LR are grouped into one param group.
    Args:
        model       (nn.Module): The model whose parameters are being grouped.
        alpha       (float):     Base (maximum) learning rate applied to the first layer.
        layer_decay (float):     Multiplicative decay factor per layer, typically in (0, 1).
        num_layers  (int):       Total number of transformer layers. Layer IDs are clamped to this.
    Returns:
        param_groups (list[dict]): List of dicts with keys 'params', 'lr', 'weight_decay'.
                                   Ready to pass directly to any torch.optim optimizer.
    """
    group_dict = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = min(get_layer_id(name), num_layers)
        lr = alpha * (layer_decay ** layer_id)
        group_dict[lr].append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": 1e-4}
        for lr, params in group_dict.items()
    ]
    print(blue("LAYER-WISE:"))
    for layer, g in enumerate(param_groups):
        print(f'learning rate for layer {layer+1}: {g["lr"]}')
    return param_groups


def valley_llrd(model: nn.Module, alpha: float, layer_decay: float, num_layers: int) -> list[dict]:
    """
    U-shaped schedule: high LR at both ends, lowest LR in the middle layers.
    Concretely for a 12-layer ViT:
        Layers  1–4  → alpha                         (strong, early features)
        Layers  5–8  → alpha * layer_decay            (medium, mid-level features)
        Layers 9–12  → alpha * layer_decay²           (low, task-specific head)
    This is the "valley" variant you asked for — soft on both ends of the network.
    Args:
        model       (nn.Module): The model whose parameters are being grouped.
        alpha       (float):     Peak learning rate (applied to early + late layers).
        layer_decay (float):     Decay per band step, typically in (0, 1), e.g. 0.65.
        num_layers  (int):       Total number of transformer layers.
    Returns:
        param_groups (list[dict]): List of dicts with keys 'params', 'lr', 'weight_decay'.
    """
    group_dict = defaultdict(list)
    band       = num_layers // 3  # ≈ 4 layers per band for a 12-layer ViT

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = min(get_layer_id(name), num_layers)

        if layer_id <= band:
            lr = alpha # early: high
        elif layer_id <= 2 * band:
            lr = alpha * layer_decay# mid:medium
        else:
            lr = alpha * (layer_decay ** 2) # late:  low

        group_dict[lr].append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": 1e-4}
        for lr, params in group_dict.items()
    ]
    print(blue("LAYER-WISE:"))
    for layer, g in enumerate(param_groups):
        print(f'learning rate for layer {layer+1}: {g["lr"]}')
    return param_groups