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
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS  = 0
CHECKPOINT_DIR = "models"
LOG_DIR        = "output/runs"
SPLITS_JSON    = "data/person/splits.json"

# freeze backbone for first N epochs, then unfreeze
FREEZE_EPOCHS  = 5
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

    # ── Optimiser + Scheduler ─────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    # ── Tensorboard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ── Training loop ─────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        # unfreeze backbone after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(model)
            # reset optimiser with lower LR for full fine-tuning
            optimizer = optim.AdamW(
                model.parameters(), lr=LR * 0.1,
                weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS, eta_min=1e-6
            )

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

        train_loss, train_cls, train_loc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch)

        val_loss, val_cls, val_loc = validate(
            model, val_loader, criterion)

        scheduler.step()
        elapsed = time.time() - start

        print(f"  Train loss : {train_loss:.4f} "
              f"(cls: {train_cls:.4f}, loc: {train_loc:.4f})")
        print(f"  Val loss   : {val_loss:.4f} "
              f"(cls: {val_cls:.4f}, loc: {val_loc:.4f})")
        print(f"  Time       : {elapsed:.1f}s")

        # tensorboard
        writer.add_scalars("loss/total",
            {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("loss/cls",
            {"train": train_cls, "val": val_cls}, epoch)
        writer.add_scalars("loss/loc",
            {"train": train_loc, "val": val_loc}, epoch)
        writer.add_scalar("lr",
            optimizer.param_groups[0]["lr"], epoch)

        # checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, is_best)

    writer.close()
    print("\n" + "=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: models/person_detector_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()