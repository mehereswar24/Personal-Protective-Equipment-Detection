import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time
from torch.utils.tensorboard import SummaryWriter

from step1_dataset import PersonDataset, collate_fn
from step2_model import MobileNetV2SSD, AnchorGenerator, SSDLoss

# ── Config ──────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 16
NUM_EPOCHS   = 60
HEAD_LR      = 1e-3      # detection heads + extra blocks
BACKBONE_LR  = 1e-4      # MobileNetV2 backbone (10x lower)
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS  = 1
FREEZE_EPOCHS  = 5       # frozen warmup then full fine-tune
NUM_WORKERS  = 0
CHECKPOINT_DIR = "models"
LOG_DIR        = "output/runs_person"
SPLITS_JSON    = "data/person/splits.json"
PATIENCE       = 15
# ────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_splits():
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    return splits["train"], splits["val"]


def build_loaders(train_recs, val_recs):
    train_ds = PersonDataset(train_recs, augment=True)
    val_ds   = PersonDataset(val_recs,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader


def freeze_backbone(model):
    for param in model.feature1.parameters():
        param.requires_grad = False
    for param in model.feature2.parameters():
        param.requires_grad = False
    print("  Backbone frozen")


def unfreeze_backbone(model):
    for param in model.feature1.parameters():
        param.requires_grad = True
    for param in model.feature2.parameters():
        param.requires_grad = True
    print("  Backbone unfrozen — full fine-tuning")


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(DEVICE)

        optimizer.zero_grad()
        cls_logits, box_preds = model(images)
        loss, cls_l, loc_l = criterion(cls_logits, box_preds, targets)

        loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        cls_loss_sum += cls_l.item()
        loc_loss_sum += loc_l.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"    Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"(cls: {cls_l.item():.4f}, loc: {loc_l.item():.4f})")

    n = len(loader)
    return total_loss/n, cls_loss_sum/n, loc_loss_sum/n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0

    for images, targets in loader:
        images = images.to(DEVICE)
        cls_logits, box_preds = model(images)
        loss, cls_l, loc_l = criterion(cls_logits, box_preds, targets)
        total_loss   += loss.item()
        cls_loss_sum += cls_l.item()
        loc_loss_sum += loc_l.item()

    n = len(loader)
    return total_loss/n, cls_loss_sum/n, loc_loss_sum/n


def save_checkpoint(model, optimizer, epoch, val_loss, is_best=False):
    state = {
        "epoch"      : epoch,
        "model"      : model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "val_loss"   : val_loss,
    }
    path = os.path.join(CHECKPOINT_DIR, "person_detector_last.pth")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "person_detector_best.pth")
        torch.save(state, best_path)
        print(f"  *** New best model saved (val_loss: {val_loss:.4f}) ***")


def _build_optimizer(model, backbone_frozen):
    backbone_params = list(model.feature1.parameters()) + \
                      list(model.feature2.parameters())
    head_params     = [p for n, p in model.named_parameters()
                       if not n.startswith(("feature1.", "feature2."))]
    return optim.AdamW(
        [
            {"params": backbone_params,
             "lr": 0.0 if backbone_frozen else BACKBONE_LR,
             "name": "backbone"},
            {"params": head_params,
             "lr": HEAD_LR,
             "name": "heads"},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _set_lr(optimizer, epoch):
    import math
    if epoch < WARMUP_EPOCHS:
        factor = (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        factor = 0.5 * (1 + math.cos(math.pi * progress)) * 0.99 + 0.01
    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = factor * BACKBONE_LR if group["lr"] > 0 else 0.0
        else:
            group["lr"] = factor * HEAD_LR


def main():
    print("=" * 60)
    print(f"Training person detector on {DEVICE}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    train_recs, val_recs = load_splits()
    train_loader, val_loader = build_loaders(train_recs, val_recs)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    model = MobileNetV2SSD(num_classes=2).to(DEVICE)
    freeze_backbone(model)

    # ── Anchors + Loss ────────────────────────────────────────
    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    criterion  = SSDLoss(anchors, device=str(DEVICE))

    # ── Optimiser ─────────────────────────────────────────────
    optimizer = _build_optimizer(model, backbone_frozen=True)

    # ── Tensorboard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ── Training loop ─────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(model)
            for group in optimizer.param_groups:
                if group["name"] == "backbone":
                    group["lr"] = BACKBONE_LR

        _set_lr(optimizer, epoch - 1)

        lrs = " / ".join(f"{g['name']}={g['lr']:.2e}"
                         for g in optimizer.param_groups)
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} (lr: {lrs})")

        train_loss, train_cls, train_loc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)

        val_loss, val_cls, val_loc = validate(
            model, val_loader, criterion)

        elapsed = time.time() - start

        print(f"  Train loss : {train_loss:.4f} "
              f"(cls: {train_cls:.4f}, loc: {train_loc:.4f})")
        print(f"  Val loss   : {val_loss:.4f} "
              f"(cls: {val_cls:.4f}, loc: {val_loc:.4f})")
        print(f"  Time       : {elapsed:.1f}s")

        writer.add_scalars("loss/total",
            {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("loss/cls",
            {"train": train_cls, "val": val_cls}, epoch)
        writer.add_scalars("loss/loc",
            {"train": train_loc, "val": val_loc}, epoch)
        for g in optimizer.param_groups:
            writer.add_scalar(f"lr/{g['name']}", g["lr"], epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
        save_checkpoint(model, optimizer, epoch, val_loss, is_best)

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    writer.close()
    print("\n" + "=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: models/person_detector_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()