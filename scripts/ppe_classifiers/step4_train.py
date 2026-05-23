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
LR             = 5e-3
WEIGHT_DECAY   = 5e-4
FREEZE_EPOCHS  = 8
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

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    best_val_loss = float("inf")
    no_improve    = 0
    PATIENCE      = 15

    for epoch in range(1, NUM_EPOCHS+1):
        t0 = time.time()

        # unfreeze backbone after freeze epochs
        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(model)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LR*0.1, weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS-FREEZE_EPOCHS, eta_min=1e-6
            )

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

        train_loss, train_cls, train_loc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)

        val_loss, val_cls, val_loc = validate(
            model, val_loader, criterion)

        scheduler.step()
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