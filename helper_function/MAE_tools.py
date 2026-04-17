from typing import Tuple

import torch


def random_masking(B: int, N: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Generates a random boolean mask. True = Masked (hidden), False = Visible.
    """
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    len_keep = int(N * (1 - mask_ratio))

    mask = torch.ones(B, N, device=device)
    mask[:, :len_keep] = 0

    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.bool()

def patchify(imgs: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts 2D images (B, C, H, W) into a flat sequence of patches (B, N, patch_dim)
    so we can calculate MSE loss against the Transformer's output.
    """
    p_h, p_w = patch_size
    B, C, H, W = imgs.shape
    assert H % p_h == 0 and W % p_w == 0, f"Image size ({H}x{W}) not divisible by patch size ({p_h}x{p_w})"
    x = imgs.reshape(B, C, H // p_h, p_h, W // p_w, p_w)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, (H // p_h) * (W // p_w), C * p_h * p_w)
    return x
