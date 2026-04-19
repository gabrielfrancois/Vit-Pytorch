import torch
from PIL import Image
from torchvision import transforms as T

from src.models.dynamicViT import DynamicVisionTransformer



IMAGE_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# 1. Load the public Student checkpoint shipped with this repo.
ckpt_path = "checkpoints/public/imagenet/baseline_noSSSL_noREPA/student_bestd_model96.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2. Rebuild the exact architecture the checkpoint was trained with.
#    Hyperparameters are persisted inside the .pth by train_student.py.
student = DynamicVisionTransformer(**ckpt["hyperparameters"])
student.load_state_dict(
    {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()},
    strict=False,
)
student.eval()

# 3. Run on a new image.
img = Image.open("data/images/sample_7.png").convert("RGB")
x   = preprocess(img).unsqueeze(0)                 # (1, 3, 128, 128)

with torch.inference_mode():
    logits, _feats, keep_masks, _scores = student(x)

probs   = torch.softmax(logits, dim=-1)
top5_p, top5_i = probs.topk(5)
print(list(zip(top5_i[0].tolist(), top5_p[0].tolist())))
print("Done!")