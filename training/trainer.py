# training/trainer.py
import os
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from models.vision_transformer import VisionTransformer

class ViTTrainer:
    def __init__(self, model_params, train_params, device=None, checkpoint_dir="checkpoints", plot_dir="plots"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionTransformer(**model_params).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=train_params.get("label_smoothing", 0.0))
        self.optimizer = AdamW(self.model.parameters(),
                               lr=train_params.get("lr", 1e-3),
                               weight_decay=train_params.get("weight_decay", 1e-4))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_params.get("step_size", 20),
            gamma=train_params.get("gamma", 0.1)
        )
        self.writer = SummaryWriter(train_params.get("log_dir", "runs/ViTTrainer"))
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.lrs = []

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(loader, desc="Training")
        for imgs, labels in loop:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        return avg_loss, accuracy

    def validate_one_epoch(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            loop = tqdm(loader, desc="Validation")
            for imgs, labels in loop:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        cm = confusion_matrix(all_labels, all_preds)

        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)

        return avg_loss, accuracy, cm

    def step_scheduler(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        self.lrs.append(lr)
        return lr

    def save_checkpoint(self, name):
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"[Checkpoint saved] {path}")

    def test_model(self, loader, checkpoint_name):
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        start_time = time.time()
        with torch.no_grad():
            loop = tqdm(loader, desc="Testing")
            for imgs, labels in loop:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / total
        accuracy = 100 * correct / total
        cm = confusion_matrix(all_labels, all_preds)
        print(f"[Test] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # save confusion matrix
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground truth")
        plt.title(f"Test Confusion Matrix ({checkpoint_name})")
        plt.savefig(os.path.join(self.plot_dir, f"{checkpoint_name}_cm.png"))
        plt.close()

        # classification report
        report = classification_report(all_labels, all_preds, zero_division=0, digits=3)
        with open(os.path.join(self.plot_dir, f"{checkpoint_name}_metrics.txt"), "w") as f:
            f.write(report)
        return avg_loss, accuracy, cm, report
