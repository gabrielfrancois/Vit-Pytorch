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
from torch.amp import autocast, GradScaler
from models.vision_transformer import VisionTransformer
import glob

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
        self.cm_max = None

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.lrs = []
        self.scaler = GradScaler()

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(loader, desc="Training")
        for imgs, labels in loop:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
        if self.cm_max is None or accuracy > (self.val_accs[-1] if self.val_accs else 0):
            self.cm_max = cm

        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        torch.cuda.empty_cache()
        return avg_loss, accuracy, cm

    def step_scheduler(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        self.lrs.append(lr)
        return lr



    def save_checkpoint(self, name, epoch, is_best=False, keep_last=3):
        """
        Sauvegarde un checkpoint et garde les N derniers.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Nom du fichier
        if is_best:
            filename = f"{name}_best.pth"
        else:
            filename = f"{name}_epoch{epoch}.pth"

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "lrs": self.lrs,
        }, path)
        print(f"[Checkpoint saved] {path}")

        # Nettoyage automatique : garder seulement les N derniers
        if not is_best:
            pattern = os.path.join(self.checkpoint_dir, f"{name}_epoch*.pth")
            all_ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
            if len(all_ckpts) > keep_last:
                for ckpt in all_ckpts[:-keep_last]:
                    os.remove(ckpt)
                    print(f"[Checkpoint deleted] {ckpt}")


    def load_checkpoint(self, name, best=False):
        """
        Charge le dernier checkpoint sauvegardé (ou le meilleur si best=True).
        """
        if best:
            path = os.path.join(self.checkpoint_dir, f"{name}_best.pth")
        else:
            # On récupère le dernier epoch sauvegardé
            ckpts = sorted(
                glob.glob(os.path.join(self.checkpoint_dir, f"{name}_epoch*.pth")),
                key=os.path.getmtime
            )
            if not ckpts:
                print("[No checkpoint found]")
                return None, 0
            path = ckpts[-1]

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_accs = checkpoint.get("train_accs", [])
        self.val_accs = checkpoint.get("val_accs", [])
        self.lrs = checkpoint.get("lrs", [])
        self.cm_max = checkpoint.get("cm_max", None)
        epoch = checkpoint.get("epoch", 0)
        print(f"[Checkpoint loaded] {path} (epoch {epoch})")
        return checkpoint, epoch


    def test_model(self, loader, checkpoint_name):
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

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
        self.cm_max =cm
        print(f"[Test] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # save confusion matrix (ne marche pas sur des données réduites de tests imagenet mais ce n'est pas grave)
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
        torch.cuda.empty_cache()
        return avg_loss, accuracy, cm, report
