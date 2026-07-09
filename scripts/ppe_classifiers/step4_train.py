import torch
import torch.optim as optim
import os
import time
import sys
sys.path.append("scripts/ppe_classifiers")

from step2_model import PPEDetector, AnchorGenerator, SSDLoss, CLASSES, NUM_CLASSES
from step3_dataset import build_loaders

# ── Config ──────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 16
NUM_EPOCHS     = 80
# AdamW with sub-1e-3 LR is the standard for SSD-MobileNet finetuning.
HEAD_LR        = 1e-3      # detection heads + extra blocks
BACKBONE_LR    = 1e-4      # MobileNetV2 backbone (10× lower)
WEIGHT_DECAY   = 5e-4
FREEZE_EPOCHS  = 5         # frozen warmup, then unfreeze
WARMUP_EPOCHS  = 1         # linear LR warmup at the start
CHECKPOINT_DIR = "models"
LOG_EVERY      = 10
# ────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def freeze_backbone(model):
    for p in model.feature1.parameters(): p.requires_grad = False
    for p in model.feature2.parameters(): p.requires_grad = False
    print("  Backbone frozen")


def unfreeze_backbone(model):
    for p in model.feature1.parameters(): p.requires_grad = True
    for p in model.feature2.parameters(): p.requires_grad = True
    print("  Backbone unfrozen")


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total = cls_sum = loc_sum = 0.0

    for i, (images, targets) in enumerate(loader):
        images = images.to(DEVICE)
        optimizer.zero_grad()
        cls_logits, box_preds = model(images)
        loss, cls_l, loc_l   = criterion(cls_logits, box_preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total   += loss.item()
        cls_sum += cls_l.item()
        loc_sum += loc_l.item()

        if (i+1) % LOG_EVERY == 0:
            print(f"    Epoch {epoch} | Batch {i+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"(cls: {cls_l.item():.4f} "
                  f"loc: {loc_l.item():.4f})")

    n = len(loader)
    return total/n, cls_sum/n, loc_sum/n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total = cls_sum = loc_sum = 0.0
    for images, targets in loader:
        images = images.to(DEVICE)
        cls_logits, box_preds = model(images)
        loss, cls_l, loc_l   = criterion(cls_logits, box_preds, targets)
        total   += loss.item()
        cls_sum += cls_l.item()
        loc_sum += loc_l.item()
    n = len(loader)
    return total/n, cls_sum/n, loc_sum/n


def save_checkpoint(model, optimizer, epoch, val_loss, is_best):
    state = {
        "epoch"    : epoch,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss" : val_loss,
        "classes"  : CLASSES,
    }
    torch.save(state, f"{CHECKPOINT_DIR}/ppe_detector_last.pth")
    if is_best:
        torch.save(state, f"{CHECKPOINT_DIR}/ppe_detector_best.pth")
        print(f"  *** Best model saved (val_loss: {val_loss:.4f}) ***")


def _build_optimizer(model, backbone_frozen):
    """Two param groups: heads (HEAD_LR) and backbone (BACKBONE_LR).
    When the backbone is frozen, its param group is included with
    requires_grad=False so the optimizer just keeps state for it."""
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
    """Linear warmup for WARMUP_EPOCHS, then cosine to eta_min over
    the remaining epochs. Backbone LR is scaled to BACKBONE_LR and
    head LR to HEAD_LR (their respective targets) by the same factor."""
    if epoch < WARMUP_EPOCHS:
        factor = (epoch + 1) / WARMUP_EPOCHS
    else:
        import math
        progress = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        # Cosine decay from 1.0 → ~0.01
        factor = 0.5 * (1 + math.cos(math.pi * progress)) * 0.99 + 0.01

    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = factor * BACKBONE_LR if group["lr"] > 0 else 0.0
        else:
            group["lr"] = factor * HEAD_LR


def main():
    print("=" * 60)
    print(f"Training PPE detector on {DEVICE}")
    print(f"Classes: {CLASSES}")
    print("=" * 60)

    train_loader, val_loader, _ = build_loaders(BATCH_SIZE)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    model = PPEDetector(num_classes=NUM_CLASSES).to(DEVICE)
    freeze_backbone(model)

    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    criterion  = SSDLoss(anchors, device=str(DEVICE))

    optimizer = _build_optimizer(model, backbone_frozen=True)

    best_val_loss = float("inf")
    no_improve    = 0
    PATIENCE      = 15

    for epoch in range(1, NUM_EPOCHS+1):
        t0 = time.time()

        # Unfreeze backbone: just flip requires_grad and let its LR
        # group come alive at the cosine-scaled BACKBONE_LR.
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

        elapsed = time.time() - t0

        print(f"  Train : {train_loss:.4f} "
              f"(cls:{train_cls:.4f} loc:{train_loc:.4f})")
        print(f"  Val   : {val_loss:.4f} "
              f"(cls:{val_cls:.4f} loc:{val_loc:.4f})")
        print(f"  Time  : {elapsed:.1f}s")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1

        save_checkpoint(model, optimizer, epoch, val_loss, is_best)

        # early stopping
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print("\n" + "=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: models/ppe_detector_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()