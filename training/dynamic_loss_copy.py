import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_function.print import *
from configs.train_cifar10 import *


class DynamicViTLoss(nn.Module):
    def __init__(self,target_ratios, lambda_kl=0.5, lambda_ratio=2.0, lambda_distill=0.5):
        super().__init__()
        self.lambda_kl = lambda_kl         # Weight for distilling teacher knowledge
        self.lambda_ratio = lambda_ratio   # Weight for enforcing sparsity
        self.lambda_distill = lambda_distill # weight for mimic the teacher model
        self.target_ratios = target_ratios # Keep 70% of tokens

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, labels, student_feats, teacher_feats, all_masks):
        """
        Calcul de la loss dynamique pour DynamicViT.
        """
        device = student_feats.device

        # S'assurer que tous les masques sont sur le bon device
        all_masks = [mask.to(device) for mask in all_masks]
        final_mask = all_masks[-1]

        # --- Classification Loss ---
        loss_cls = self.ce_loss(student_logits, labels)

        # --- KL Divergence avec le Teacher ---
        loss_kl = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )

        # --- Distillation des features ---
        token_diff = (student_feats - teacher_feats).pow(2).sum(dim=-1)  # sum sur D_model
        masked_diff = token_diff * final_mask
        loss_distill = masked_diff.sum() / (final_mask.sum() + 1e-6)

        # --- Ratio Loss pour chaque étape de pruning ---
        loss_ratio = 0.0
        for i, mask_s in enumerate(all_masks):
            actual_ratio = mask_s.float().mean(dim=1)  # mean sur tokens, size=(B)
            target = self.target_ratios[i]
            loss_ratio += ((target - actual_ratio) ** 2).mean()  # mean sur batch

        # --- Loss totale ---
        total_loss = (
            loss_cls +
            self.lambda_kl * loss_kl +
            self.lambda_distill * loss_distill +
            self.lambda_ratio * loss_ratio
        )

        metrics = {
            "cls": loss_cls.item(),
            "distill": loss_distill.item(),
            "ratio": loss_ratio.item(),
            "kl": loss_kl.item()
        }

        return total_loss, metrics
