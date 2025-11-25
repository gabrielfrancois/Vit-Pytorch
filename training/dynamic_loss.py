import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.train_cifar10 import *
from helper_function.print import *


class DynamicViTLoss(nn.Module):
    def __init__(
        self, target_ratios, lambda_kl=0.5, lambda_ratio=2.0, lambda_distill=0.5
    ):
        super().__init__()
        self.lambda_kl = lambda_kl  # Weight for distilling teacher knowledge
        self.lambda_ratio = lambda_ratio  # Weight for enforcing sparsity
        self.lambda_distill = lambda_distill  # weight for mimic the teacher model
        self.target_ratios = target_ratios  # Keep 70% of tokens

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits,
        teacher_logits,
        labels,
        student_feats,
        teacher_feats,
        all_masks,
    ):
        """
        input:
        ----------------------------------
            - student_logits: (B, n_classes) : Output from the DynamicViT
            - teacher_logits: (B, n_classes) : Output from the frozen Teacher ViT
            - labels: (B) : Ground truth labels
            - student_feats : (B, N, d_model)  : Feature vectors (t_i) of student after the last block
            - teacher_feats : (B, N, d_model) : Feature vectors (t_i')of teacher after the last block
            - all_masks: List of tensors [(B, N), ...]. Binary masks from each pruning stage.
                        Thus, the last item in the list corresponds to D^{b, S} (final mask).
        output:
            - total_loss: float
            - dict with cross entropy loss, distill loss and ratio loss
        ----------------------------------

        """

        # Classification Loss (Student vs Ground Truth)
        loss_cls = self.ce_loss(student_logits, labels)

        # KL divergence
        loss_kl = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )

        # Uses the FINAL mask (D at last stage)
        final_mask = all_masks[-1]
        token_diff = (
            (student_feats - teacher_feats).pow(2).sum(dim=-1)
        )  # Sum over D_model dim
        masked_diff = token_diff * final_mask
        loss_distill = masked_diff.sum() / (
            final_mask.sum() + 1e-6
        )  # +1e-6 to avoid the cancel out of the behind the divison

        # Ratio Loss
        loss_ratio = 0.0
        for i, mask_s in enumerate(all_masks):
            # mask_s = D^{b, s} of size (B, N)
            # Calculate actual ratio: (Sum of 1s) / N
            actual_ratio = mask_s.mean(dim=1)  # Mean over tokens -> size = (B)
            target = self.target_ratios[i]  # the rho^(s) of the paper

            # MSE: (Actual - Target)^2
            loss_ratio += (target - actual_ratio).pow(2).mean()  # Mean over Batch

        total_loss = (
            loss_cls
            + (self.lambda_kl * loss_kl)
            + (self.lambda_distill * loss_distill)
            + (self.lambda_ratio * loss_ratio)
        )
        return total_loss, {
            "cls": loss_cls.item(),
            "distill": loss_distill.item(),
            "ratio": loss_ratio.item(),
            "kl": loss_kl.item(),
        }
