import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_function.print import *
from configs.train_cifar10 import *


class DynamicViTLoss(nn.Module):
    def __init__(self, lambda_kl=0.5, lambda_ratio=2.0, target_keep_rate=0.7):
        super().__init__()
        self.lambda_kl = lambda_kl         # Weight for distilling teacher knowledge
        self.lambda_ratio = lambda_ratio   # Weight for enforcing sparsity
        self.lambda_distill = lambda_distill # weight for mimic the teacher model
        self.target_keep_rate = target_keep_rate # Keep 70% of tokens
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, labels, pred_scores):
        """
        student_logits: (B, n_classes) : Output from the DynamicViT
        teacher_logits: (B, n_classes) : Output from the frozen Teacher ViT
        labels: (B) : Ground truth labels
        pred_scores: List of (B, N) tensors : The 'keep' probabilities from PredictorLG
        """
        
        # Classification Loss (Student vs Ground Truth)
        loss_cls = self.ce_loss(student_logits, labels)

        # Distillation Loss (Student vs Teacher) 
        # We want the student to ape the teacher's probability distribution
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )

        # Ratio Loss (Sparsity Constraint) [cite: 188]
        # We want the average keep rate to match target_keep_rate
        loss_ratio = 0.0
        for score in pred_scores:
            # score is (B, N) probability of keeping
            current_ratio = score.mean()
            loss_ratio += self.mse_loss(current_ratio, torch.tensor(self.target_keep_rate, device=score.device))

        # Combine them 
        total_loss = loss_cls + (self.lambda_distill * distillation_loss) + (self.lambda_ratio * loss_ratio)
        
        return total_loss, {"cls": loss_cls.item(), "distill": distillation_loss.item(), "ratio": loss_ratio.item()}