import os
from typing import Mapping

import torch
from torch import nn

from .print import *

def verbose_load(model: nn.Module(), state_dict: Mapping[str, torch.Tensor]):
    """
    Load a state_dict into a model with strict=False and print a summary of missing
    and unexpected keys. Automatically removes the "_orig_mod." prefix (from torch.compile).
    
    Args:
        model (nn.Module): Target model to load weights into.
        state_dict (Mapping[str, torch.Tensor]): Mapping of parameter names to tensors.
    """
    # print("Remove _orig_mod. from the .compile to enable the weight loading")
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    load_results = model.load_state_dict(clean_state_dict, strict=False)
    
    print("\n[Weight Loading Summary]")
    
    # In the class, but NOT in the checkpoint.
    missing_keys = load_results.missing_keys
    if missing_keys:
        print(orange("Missing Keys (In Model, Not in Checkpoint -> Randomly Initialized):"))
        for key in missing_keys:
            print(f"   - {key}")
            
    # In the checkpoint, but NOT in the class.
    unexpected_keys = load_results.unexpected_keys
    if unexpected_keys:
        print(orange("Unexpected Keys (In Checkpoint, Not in Model -> Dropped):"))
        for key in unexpected_keys:
            print(f"   - {key}")
            
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(clean_state_dict.keys()) & model_keys
    print(green(f"\n Successfully loaded {len(loaded_keys)} out of {len(model_keys)} parameter tensors."))

def print_layer_shapes(state_dict):
    print("\n[Model Architecture Shapes]")
    for layer_name, weight_tensor in state_dict.items():
        print(f"{layer_name} | Shape: {list(weight_tensor.shape)}")