# grape_cls.py
import os, random, json, argparse, glob
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# ============ 常量 ============
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============ 工具函数 ============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_images(root: str) -> Tuple[List[str], List[int], Dict[str,int]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not classes:
        raise RuntimeError(f"未发现类别子目录，请检查: {root}")
    class_to_idx = {c:i for i,c in enumerate(classes)}

    paths, labels = [], []
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in glob.glob(os.path.join(cdir, "**", "*"), recursive=True):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(fn)
                labels.append(class_to_idx[c])

    if len(paths) == 0:
        raise RuntimeError(f"未在 {root} 找到图像文件")
    return paths, labels, class_to_idx

class AlbumentationsDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        y = self.labels[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            out = self.transform(image=img)
            img = out["image"]
        return img, y, os.path.basename(p)

# ============ 数据增强 ============
def build_transforms(img_size: int = 512):
    train_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.MotionBlur(p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    return train_tf, val_tf

# ============ 模型 ============
def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower()
    if model_name in ["efficientnet_v2_s", "effv2s", "efficientnet"]:
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_feats, num_classes)
        )
    elif model_name in ["convnext_tiny", "convnext_t"]:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")
    return model

# ============ 损失权重 ============
def compute_class_weights(y: List[int], num_classes: int, power: float = 1.0):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts) ** power
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def build_sampler(labels: List[int]):
    class_count = np.bincount(labels)
    class_count[class_count == 0] = 1
    class_weights = 1.0 / class_count
    sample_weights = np.array([class_weights[y] for y in labels])
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ============ 保存模型 ============
def save_checkpoint(path: str, model: nn.Module, class_to_idx: Dict[str,int], img_size: int, model_name: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "model_name": model_name,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    }
    torch.save(payload, path)

# ============ 验证 ============
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, np.array(y_true), np.array(y_pred)

# ============ 参数 ============
def get_args():
    ap = argparse.ArgumentParser(description="Plant disease classifier (Albumentations 2.x)")
    ap.add_argument("--data_dir", type=str, required=True, help="分类数据根目录")
    ap.add_argument("--out_dir", type=str, default="./outputs", help="模型输出目录")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--model", type=str, default="efficientnet_v2_s",
                    choices=["efficientnet_v2_s", "convnext_tiny"])
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ============ 主函数 ============
def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device} | CUDA? {torch.cuda.is_available()}")

    paths, labels, class_to_idx = find_images(args.data_dir)
    classes = list(class_to_idx.keys())
    print(f"发现 {len(classes)} 个类别, 总图像 {len(paths)} 张")

    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=args.val_ratio, random_state=args.seed, stratify=labels
    )

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = AlbumentationsDataset(X_train, y_train, transform=train_tf)
    val_ds   = AlbumentationsDataset(X_val,   y_val,   transform=val_tf)

    if args.use_weighted_sampler:
        sampler = build_sampler(y_train)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=len(classes), pretrained=not args.no_pretrained)
    model.to(device)

    class_weights = compute_class_weights(y_train, num_classes=len(classes), power=1.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs)
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        return 1.0
    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1 = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        warmup.step()
        cosine.step()

        acc, macro_f1, y_true, y_pred = evaluate(model, val_loader, device)
        print(f"Val Acc={acc:.4f} | Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            save_checkpoint(ckpt_path, model, class_to_idx, args.img_size, args.model)
            print(f"💾 保存最优权重到 {ckpt_path} (macro-F1={best_f1:.4f})")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        np.savetxt(os.path.join(args.out_dir, f"cm_epoch{epoch}.csv"), cm, fmt="%d", delimiter=",")

    print("训练完成。最优Macro-F1:", best_f1)

# ============ 推理 ============
@torch.no_grad()
def load_model(ckpt_path: str, device=None):
    payload = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = payload["class_to_idx"]
    model = build_model(payload["model_name"], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, payload

@torch.no_grad()
def predict_image(ckpt_path: str, image_path: str, topk: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(ckpt_path, device)
    img_size = meta["img_size"]
    mean, std = meta["mean"], meta["std"]
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tf(image=img)["image"].unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:topk]
    result = [(idx_to_class[i], float(probs[i])) for i in top_idx]
    return result

def export_onnx(ckpt_path: str, onnx_path: str = "model.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(ckpt_path, device)
    model.eval()
    img_size = meta["img_size"]
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )
    print(f"ONNX 导出完成: {onnx_path}")

if __name__ == "__main__":
    main()
