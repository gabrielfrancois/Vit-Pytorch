from data.imagenet_loader import load_imagenet1k
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _, _ = load_imagenet1k(batch_size=64)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

print(f"Images dtype: {images.dtype}, device: {images.device}")

# Essaie juste un passage dans un modèle bidon pour vérifier la mémoire
images = images.to(device)
print("Batch transféré sur GPU avec succès.")
