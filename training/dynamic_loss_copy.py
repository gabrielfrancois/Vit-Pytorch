import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_function.print import *
from configs.train_imagenet1k import *



class DynamicViTLoss(nn.Module):
    def __init__(self,target_ratios, lambda_kl, lambda_ratio, lambda_distill):
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

        # Petite optimisation (on utilise F.log_softmax et log _target= True vs F.softmax)
        loss_kl = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.log_softmax(teacher_logits.detach(), dim=1),
            reduction='batchmean',
            log_target=True
        )


        # --- Distillation des features ---
        token_diff = (student_feats - teacher_feats).pow(2).sum(dim=-1)  # sum sur D_model
        masked_diff = token_diff * final_mask
        loss_distill = masked_diff.sum() / (final_mask.sum() + 1e-6)

        # --- Ratio Loss pour chaque étape de pruning (version vectorisée) ---
        all_masks_tensor = torch.stack(all_masks, dim=0)  # (T, B, N)
        actual_ratios = all_masks_tensor.float().mean(dim=-1)  # (T, B)
        targets = torch.tensor(self.target_ratios, device=actual_ratios.device).unsqueeze(1)  # (T,1)
        loss_ratio = ((targets - actual_ratios)**2).mean(dim=1).sum()


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
