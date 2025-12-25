import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_function.print import *
from configs.train_imagenet1k import *
from typing import List, Tuple, Dict


class DynamicViTLoss(nn.Module):
    def __init__(
        self,
        target_ratios: List[float],
        lambda_class: float,
        lambda_kl: float,
        lambda_ratio: float,
        lambda_distill: float
    ) -> None:
        """
        Dynamic loss function for DynamicViT student model.

        Combines classification loss, KL divergence with teacher, feature distillation,
        and sparsity enforcement via token pruning masks.

        Args:
            target_ratios: List of target keep ratios for each pruning layer.
            lambda_class: Weight for the classification loss.
            lambda_kl: Weight for the KL divergence loss between student and teacher logits.
            lambda_ratio: Weight for enforcing sparsity through pruning.
            lambda_distill: Weight for distilling features from teacher to student.
        """
        super().__init__()
        self.lambda_class = lambda_class
        self.lambda_kl = lambda_kl
        self.lambda_ratio = lambda_ratio
        self.lambda_distill = lambda_distill
        self.target_ratios = target_ratios

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        all_masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the dynamic loss for a batch of student outputs.

        Args:
            student_logits: Student model class logits (B, n_classes).
            teacher_logits: Teacher model class logits (B, n_classes), detached.
            labels: Ground-truth labels (B,).
            student_feats: Features from student model (B, N, d_model).
            teacher_feats: Features from teacher model (B, N, d_model).
            all_masks: List of binary pruning masks applied at each pruned layer [(B, N), ...].

        Returns:
            total_loss: Weighted sum of classification, KL, distillation, and ratio losses.
            metrics: Dictionary with individual losses:
                - "cls": Classification loss
                - "distill": Feature distillation loss
                - "ratio": Sparsity (pruning) loss
                - "kl": KL divergence with teacher logits
        """
        device = student_feats.device
        all_masks = [mask.to(device) for mask in all_masks]
        final_mask = all_masks[-1]

        # Classification loss
        loss_cls = self.ce_loss(student_logits, labels)

        # KL divergence with teacher
        loss_kl = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.log_softmax(teacher_logits.detach(), dim=1),
            reduction='batchmean',
            log_target=True
        )

        # Feature distillation loss
        token_diff = (student_feats - teacher_feats).pow(2).sum(dim=-1)  # sum over d_model
        masked_diff = token_diff * final_mask
        loss_distill = masked_diff.sum() / (final_mask.sum() + 1e-6)

        # Ratio loss for each pruning step
        all_masks_tensor = torch.stack(all_masks, dim=0)  # (T, B, N)
        actual_ratios = all_masks_tensor.float().mean(dim=-1)  # (T, B)
        targets = torch.tensor(self.target_ratios, device=actual_ratios.device).unsqueeze(1)  # (T,1)
        loss_ratio = ((targets - actual_ratios)**2).mean(dim=1).sum()

        # Total loss
        total_loss = (
            self.lambda_class * loss_cls +
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
