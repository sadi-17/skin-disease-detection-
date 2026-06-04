# ============================================================================
#  SELF-SUPERVISED LEARNING FOR SKIN DISEASE CLASSIFICATION
#  SimCLR pre-training + supervised fine-tuning on DermnetClinical (23 classes)
#
#  >>> SINGLE KAGGLE CELL <<<  | PyTorch | GPU-ready | IEEE-publication ready
#
#  Pipeline:
#     1) SimCLR self-supervised pre-training (NT-Xent contrastive loss)
#     2) Supervised fine-tuning (projection head removed, classifier added)
#     3) Full IEEE-level evaluation (Acc / Precision / Recall / F1 / ROC-AUC
#        + confusion matrix + loss & accuracy curves)
# ============================================================================

# =====================
# IMPORTS
# =====================
import os, gc, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (resnet50, ResNet50_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

warnings.filterwarnings("ignore")

# =====================
# CONFIG / HYPERPARAMETERS
# =====================
class CFG:
    # --- paths (edit DATA_ROOT if your dataset folder name differs) ---
    DATA_ROOT      = "/kaggle/input/dermnetclinical"
    OUT_DIR        = "/kaggle/working"

    # --- backbone: "resnet50" (default, canonical SimCLR encoder) or "efficientnet_b0" ---
    BACKBONE       = "resnet50"
    PRETRAINED_INIT= True      # ImageNet init (needs internet ON). Falls back to random if download fails.
                               # Set False for "pure" from-scratch self-supervision.

    # --- image / loader ---
    IMG_SIZE       = 224
    NUM_WORKERS    = 2

    # --- SimCLR pre-training ---
    PRETRAIN_EPOCHS= 30        # raise to 100+ for final paper runs
    PRETRAIN_BS    = 128       # SimCLR likes large batches; lower to 64 if OOM
    PRETRAIN_LR    = 1e-3
    PROJ_HIDDEN    = 512
    PROJ_DIM       = 128
    TEMPERATURE    = 0.5       # NT-Xent temperature

    # --- supervised fine-tuning ---
    FINETUNE_EPOCHS= 40
    FINETUNE_BS    = 64
    HEAD_LR        = 1e-3
    BACKBONE_LR    = 1e-4      # smaller LR for pre-trained backbone
    WEIGHT_DECAY   = 1e-4
    VAL_SPLIT      = 0.15
    EARLY_STOP_PAT = 7         # early-stopping patience (epochs)
    DROPOUT        = 0.3

    SEED           = 42
    USE_AMP        = True      # mixed precision

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

# =====================
# REPRODUCIBILITY & DEVICE
# =====================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = cfg.USE_AMP and DEVICE.type == "cuda"
print(f"Device: {DEVICE} | AMP: {USE_AMP} | torch {torch.__version__}")

# =====================
# AUGMENTATIONS  (medically safe — no geometry/colour distortion that changes diagnosis)
# =====================
# ImageNet stats (standard for transfer + works fine for clinical RGB photos)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Two-view SimCLR augmentation: stochastic but conservative for medical imagery.
simclr_aug = transforms.Compose([
    transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.6, 1.0)),  # mild crop, keeps lesion
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),  # small rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),  # mild only
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Fine-tuning train transform (light, label-preserving)
train_aug = transforms.Compose([
    transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Deterministic eval transform (resize + center crop + normalize)
eval_aug = transforms.Compose([
    transforms.Resize(int(cfg.IMG_SIZE * 1.14)),
    transforms.CenterCrop(cfg.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =====================
# DATA LOADING
# =====================
TRAIN_DIR = os.path.join(cfg.DATA_ROOT, "train")
TEST_DIR  = os.path.join(cfg.DATA_ROOT, "test")
assert os.path.isdir(TRAIN_DIR), f"train/ not found under {cfg.DATA_ROOT}. Edit CFG.DATA_ROOT."
assert os.path.isdir(TEST_DIR),  f"test/ not found under {cfg.DATA_ROOT}. Edit CFG.DATA_ROOT."

# Base ImageFolders (same deterministic sample order -> indices align across transforms)
base_train_ft   = ImageFolder(TRAIN_DIR, transform=train_aug)   # for fine-tune (train indices)
base_train_eval = ImageFolder(TRAIN_DIR, transform=eval_aug)    # for validation (val indices)
test_set        = ImageFolder(TEST_DIR,  transform=eval_aug)

CLASSES   = base_train_ft.classes
NUM_CLASS = len(CLASSES)
print(f"Found {NUM_CLASS} classes | train imgs: {len(base_train_ft)} | test imgs: {len(test_set)}")

# Stratified train/val split (on the labeled train set)
targets = np.array([s[1] for s in base_train_ft.samples])
train_idx, val_idx = train_test_split(
    np.arange(len(targets)), test_size=cfg.VAL_SPLIT,
    stratify=targets, random_state=cfg.SEED)

ft_train_subset = Subset(base_train_ft,   train_idx)   # augmented
ft_val_subset   = Subset(base_train_eval, val_idx)     # clean

# ---- SimCLR dataset: returns two independent augmented views of the same image ----
class SimCLRTwoView(Dataset):
    """Wraps an ImageFolder; uses ALL train images (labels ignored) for self-supervision."""
    def __init__(self, image_folder, transform):
        self.folder = image_folder
        self.transform = transform
    def __len__(self):
        return len(self.folder.samples)
    def __getitem__(self, i):
        path, _ = self.folder.samples[i]
        img = self.folder.loader(path)        # PIL RGB
        return self.transform(img), self.transform(img)

simclr_set = SimCLRTwoView(base_train_eval, simclr_aug)   # full train set, no labels

# ---- DataLoaders ----
simclr_loader = DataLoader(simclr_set, batch_size=cfg.PRETRAIN_BS, shuffle=True,
                           num_workers=cfg.NUM_WORKERS, pin_memory=True,
                           drop_last=True)   # drop_last keeps NT-Xent batch shape consistent
train_loader  = DataLoader(ft_train_subset, batch_size=cfg.FINETUNE_BS, shuffle=True,
                           num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader    = DataLoader(ft_val_subset, batch_size=cfg.FINETUNE_BS, shuffle=False,
                           num_workers=cfg.NUM_WORKERS, pin_memory=True)
test_loader   = DataLoader(test_set, batch_size=cfg.FINETUNE_BS, shuffle=False,
                           num_workers=cfg.NUM_WORKERS, pin_memory=True)

# =====================
# MODEL: BACKBONE + PROJECTION HEAD (SimCLR) ; BACKBONE + CLASSIFIER (fine-tune)
# =====================
def build_backbone(name, pretrained):
    """Returns (encoder_without_classifier, feature_dim). Falls back to random init if
    pretrained weights can't be downloaded (e.g. Kaggle internet OFF)."""
    try:
        if name == "resnet50":
            w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = resnet50(weights=w); feat = m.fc.in_features; m.fc = nn.Identity()
        elif name == "efficientnet_b0":
            w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = efficientnet_b0(weights=w); feat = m.classifier[1].in_features
            m.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone {name}")
    except Exception as e:
        print(f"[warn] pretrained download failed ({e}). Using random init.")
        if name == "resnet50":
            m = resnet50(weights=None); feat = m.fc.in_features; m.fc = nn.Identity()
        else:
            m = efficientnet_b0(weights=None); feat = m.classifier[1].in_features
            m.classifier = nn.Identity()
    return m, feat

class ProjectionHead(nn.Module):
    """2-layer MLP projection head used only during SimCLR pre-training."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, backbone, feat_dim, hidden, proj_dim):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(feat_dim, hidden, proj_dim)
    def forward(self, x):
        h = self.backbone(x)                 # representation
        z = self.projection(h)               # projection for contrastive loss
        return z

class ClassifierModel(nn.Module):
    """Backbone (encoder) + classification head. Projection head is dropped."""
    def __init__(self, backbone, feat_dim, num_classes, dropout):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x))

# =====================
# NT-XENT CONTRASTIVE LOSS  (normalized temperature-scaled cross entropy)
# =====================
def nt_xent_loss(z_i, z_j, temperature):
    """
    z_i, z_j : (N, D) projections of the two augmented views.
    Builds a 2N x 2N similarity matrix; for each sample its augmented counterpart
    is the single positive, all other 2N-2 samples are negatives.
    """
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)                  # (2N, D)
    z = F.normalize(z, dim=1)                          # cosine similarity space
    sim = torch.matmul(z, z.t()) / temperature         # (2N, 2N)

    # mask self-similarity on the diagonal
    diag = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag, -9e15)

    # positive index: for row k in [0,N) -> k+N ; for row k in [N,2N) -> k-N
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)

# =====================
# MODEL SUMMARY
# =====================
def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn

def model_summary(model, title):
    print("=" * 70); print(title); print("=" * 70)
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE),
                col_names=["input_size", "output_size", "num_params"], depth=2, verbose=1)
    except Exception:
        tot, trn = count_params(model)
        print(model)
        print(f"\nTotal params: {tot:,} | Trainable: {trn:,}")

# =====================
# STAGE 1 — SimCLR SELF-SUPERVISED PRE-TRAINING
# =====================
def pretrain_simclr():
    backbone, feat_dim = build_backbone(cfg.BACKBONE, cfg.PRETRAINED_INIT)
    model = SimCLRModel(backbone, feat_dim, cfg.PROJ_HIDDEN, cfg.PROJ_DIM).to(DEVICE)
    model_summary(model, f"SimCLR MODEL  (backbone={cfg.BACKBONE}, feat_dim={feat_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PRETRAIN_LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PRETRAIN_EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    print("\n>>> Stage 1: SimCLR contrastive pre-training")
    history = []
    for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
        model.train(); running = 0.0; n = 0
        pbar = tqdm(simclr_loader, desc=f"[SimCLR] Epoch {epoch}/{cfg.PRETRAIN_EPOCHS}")
        for v1, v2 in pbar:
            v1, v2 = v1.to(DEVICE, non_blocking=True), v2.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                z1, z2 = model(v1), model(v2)
                loss = nt_xent_loss(z1.float(), z2.float(), cfg.TEMPERATURE)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item() * v1.size(0); n += v1.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        ep_loss = running / n
        history.append(ep_loss)
        print(f"[SimCLR] Epoch {epoch:3d} | contrastive loss {ep_loss:.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

    # save the pre-trained ENCODER ONLY (projection head is discarded downstream)
    torch.save(model.backbone.state_dict(), os.path.join(cfg.OUT_DIR, "simclr_encoder.pt"))
    print("Saved pre-trained encoder -> simclr_encoder.pt")
    return feat_dim, history

# =====================
# FINE-TUNING HELPERS
# =====================
@torch.no_grad()
def evaluate_loader(model, loader, criterion):
    model.eval(); loss_sum, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            out = model(x); loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0); n += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / n, correct / n

# =====================
# STAGE 2 — SUPERVISED FINE-TUNING (with early stopping, LR scheduler, best-model save)
# =====================
def finetune(feat_dim):
    # rebuild backbone, load SimCLR-pretrained weights, attach classifier head
    backbone, _ = build_backbone(cfg.BACKBONE, pretrained=False)
    state = torch.load(os.path.join(cfg.OUT_DIR, "simclr_encoder.pt"), map_location="cpu")
    backbone.load_state_dict(state)
    model = ClassifierModel(backbone, feat_dim, NUM_CLASS, cfg.DROPOUT).to(DEVICE)
    model_summary(model, "FINE-TUNING MODEL  (encoder + classification head)")

    criterion = nn.CrossEntropyLoss()
    # param groups: low LR for transferred backbone, higher LR for fresh head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.BACKBONE_LR},
        {"params": model.head.parameters(),     "lr": cfg.HEAD_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(enabled=USE_AMP)

    print("\n>>> Stage 2: supervised fine-tuning")
    best_val = float("inf"); best_state = None; patience = 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.FINETUNE_EPOCHS + 1):
        model.train(); run_loss, n, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[FT] Epoch {epoch}/{cfg.FINETUNE_EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item() * x.size(0); n += x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss, tr_acc = run_loss / n, correct / n
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion)
        scheduler.step(val_loss)

        hist["train_loss"].append(tr_loss); hist["val_loss"].append(val_loss)
        hist["train_acc"].append(tr_acc);   hist["val_acc"].append(val_acc)
        print(f"[FT] Epoch {epoch:3d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # ---- best-model save + early stopping ----
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(cfg.OUT_DIR, "best_model.pt"))
            patience = 0
            print(f"      ✓ new best (val_loss={best_val:.4f}) -> saved best_model.pt")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PAT:
                print(f"      Early stopping at epoch {epoch} (no improvement {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist

# =====================
# EVALUATION (IEEE-LEVEL METRICS)
# =====================
@torch.no_grad()
def collect_predictions(model, loader):
    model.eval(); y_true, y_pred, y_prob = [], [], []
    for x, y in tqdm(loader, desc="[Eval] test set"):
        x = x.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            logits = model(x)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        y_prob.append(probs)
        y_pred.extend(probs.argmax(1).tolist())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred), np.concatenate(y_prob, axis=0)

def evaluate_and_report(model):
    y_true, y_pred, y_prob = collect_predictions(model, test_loader)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # multi-class ROC-AUC (one-vs-rest, macro)
    try:
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASS)))
        roc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception as e:
        roc = float("nan"); print(f"[warn] ROC-AUC could not be computed: {e}")

    print("\n" + "=" * 60)
    print("  FINAL TEST METRICS  (macro-averaged)")
    print("=" * 60)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall    (macro) : {rec:.4f}")
    print(f"  F1-score  (macro) : {f1:.4f}")
    print(f"  F1-score  (weighted): {f1w:.4f}")
    print(f"  ROC-AUC   (ovr,macro): {roc:.4f}")
    print("=" * 60)

    print("\nPer-class classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    # --- Confusion matrix heatmap ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASS)))
    plt.figure(figsize=(13, 11))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
    except Exception:
        plt.imshow(cm, cmap="Blues"); plt.colorbar()
        plt.xticks(range(NUM_CLASS), CLASSES, rotation=90)
        plt.yticks(range(NUM_CLASS), CLASSES)
    plt.title("Confusion Matrix — DermnetClinical (test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1_macro": f1, "f1_weighted": f1w, "roc_auc": roc}

# =====================
# CURVES (loss & accuracy)
# =====================
def plot_curves(hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ep, hist["train_loss"], "o-", label="Train loss")
    ax[0].plot(ep, hist["val_loss"],   "s-", label="Val loss")
    ax[0].set_title("Training vs Validation Loss"); ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(ep, hist["train_acc"], "o-", label="Train acc")
    ax[1].plot(ep, hist["val_acc"],   "s-", label="Val acc")
    ax[1].set_title("Training vs Validation Accuracy"); ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "training_curves.png"), dpi=150)
    plt.show()

def plot_simclr_curve(loss_hist):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(loss_hist) + 1), loss_hist, "o-")
    plt.title("SimCLR Pre-training — NT-Xent Loss")
    plt.xlabel("Epoch"); plt.ylabel("Contrastive loss"); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, "simclr_loss.png"), dpi=150)
    plt.show()

# =====================
# RUN FULL PIPELINE
# =====================
if __name__ == "__main__":
    t0 = time.time()

    # Stage 1: self-supervised pre-training
    feat_dim, simclr_hist = pretrain_simclr()
    plot_simclr_curve(simclr_hist)
    gc.collect(); torch.cuda.empty_cache()

    # Stage 2: supervised fine-tuning
    model, hist = finetune(feat_dim)
    plot_curves(hist)

    # Stage 3: evaluation
    results = evaluate_and_report(model)

    # Save a tidy results table for the paper
    pd.DataFrame([results]).to_csv(os.path.join(cfg.OUT_DIR, "final_metrics.csv"), index=False)
    print(f"\nArtifacts saved to {cfg.OUT_DIR}: "
          "simclr_encoder.pt, best_model.pt, confusion_matrix.png, "
          "training_curves.png, simclr_loss.png, final_metrics.csv")
    print(f"Total wall-clock time: {(time.time()-t0)/60:.1f} min")
