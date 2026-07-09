import torch
import torch.nn as nn
import torchvision.models as models


# ── Class definitions ────────────────────────────────────
CLASSES = [
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "gloves",
    "boots",
    "mask",
    "no_mask",
]
BACKGROUND_IDX = 0
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES  = len(CLASSES) + 1
LEGACY_NUM_CLASSES = len(CLASSES)

# Anchor count per feature-map cell. Must stay in sync with AnchorGenerator
# (2 scales x 3 aspect ratios = 6).
NUM_ANCHORS = 6


class PPEDetector(nn.Module):
    """
    SSD-style PPE detector built on MobileNetV2 backbone.
    Detects all PPE items in a single forward pass.

    Input  : [B, 3, 300, 300]
    Output :
        cls_logits : [B, total_anchors, num_classes]
        box_preds  : [B, total_anchors, 4]
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)

        self.feature1 = nn.Sequential(*backbone.features[:14])  # 96ch  19x19
        self.feature2 = nn.Sequential(*backbone.features[14:])  # 1280ch 10x10

        # ── Extra layers ──────────────────────────────────
        self.extra1 = self._extra_block(1280, 256, stride=2)    # 5x5
        self.extra2 = self._extra_block(256,  128, stride=2)    # 3x3
        self.extra3 = self._extra_block(128,  128, stride=3)    # 1x1

        # ── Detection heads ───────────────────────────────
        # 6 anchors per cell: 2 scales × 3 aspect ratios. Matches the
        # AnchorGenerator config below.
        self.num_anchors     = NUM_ANCHORS
        feature_channels     = [96, 1280, 256, 128, 128]

        self.cls_heads = nn.ModuleList([
            nn.Conv2d(ch, self.num_anchors * num_classes,
                      kernel_size=3, padding=1)
            for ch in feature_channels
        ])
        self.box_heads = nn.ModuleList([
            nn.Conv2d(ch, self.num_anchors * 4,
                      kernel_size=3, padding=1)
            for ch in feature_channels
        ])

        self._init_weights()

    def _extra_block(self, in_ch, out_ch, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, out_ch // 2),
                         num_channels=out_ch // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3,
                      stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch),
                         num_channels=out_ch),
            nn.ReLU6(inplace=True),
        )

    def _init_weights(self):
        for extra in [self.extra1, self.extra2, self.extra3]:
            for m in extra.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
        for head in self.cls_heads + self.box_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        f1 = self.feature1(x)       # [B, 96,   19, 19]
        f2 = self.feature2(f1)      # [B, 1280, 10, 10]
        f3 = self.extra1(f2)        # [B, 256,   5,  5]
        f4 = self.extra2(f3)        # [B, 128,   3,  3]
        f5 = self.extra3(f4)        # [B, 128,   1,  1]

        cls_logits, box_preds = [], []

        for feat, cls_h, box_h in zip(
                [f1,f2,f3,f4,f5], self.cls_heads, self.box_heads):
            B, _, H, W = feat.shape

            cls = cls_h(feat).permute(0,2,3,1).contiguous()
            cls = cls.view(B, -1, self.num_classes)

            box = box_h(feat).permute(0,2,3,1).contiguous()
            box = box.view(B, -1, 4)

            cls_logits.append(cls)
            box_preds.append(box)

        return (torch.cat(cls_logits, dim=1),
                torch.cat(box_preds,  dim=1))


class AnchorGenerator(nn.Module):
    """
    SSD-style anchors: at every cell of every feature map, emit
    one anchor per (scale, aspect_ratio) pair. We use two scales per
    feature map — the level's primary scale, and the geometric mean
    with the next level's scale — to cover sizes that fall between
    adjacent feature levels. With 3 aspect ratios this yields 6
    anchors per cell, matching PPEDetector.num_anchors=6.
    """

    def __init__(self):
        super().__init__()
        # (feature_map_size, primary_scale)
        levels = [
            (19, 0.10),
            (10, 0.20),
            ( 5, 0.37),
            ( 3, 0.54),
            ( 1, 0.71),
        ]
        next_scales = [s for (_, s) in levels[1:]] + [0.95]
        ratios = [1.0, 2.0, 0.5]

        self.configs = []
        for (fmap, s), s_next in zip(levels, next_scales):
            s_extra = (s * s_next) ** 0.5
            self.configs.append((fmap, [s, s_extra], ratios))

    def forward(self):
        anchors = []
        for (fmap_size, scales, ratios) in self.configs:
            for i in range(fmap_size):
                for j in range(fmap_size):
                    cx = (j + 0.5) / fmap_size
                    cy = (i + 0.5) / fmap_size
                    for scale in scales:
                        for ratio in ratios:
                            w = scale * (ratio ** 0.5)
                            h = scale / (ratio ** 0.5)
                            anchors.append([cx, cy, w, h])
        return torch.tensor(anchors, dtype=torch.float32)


class SSDLoss(nn.Module):
    """
    Focal-loss SSD criterion (RetinaNet-style).

    Focal loss already handles class imbalance by exponentially
    downweighting easy negatives, so we do NOT stack hard-negative
    mining on top of it. Per-positive normalisation matches the
    RetinaNet formulation.
    """

    def __init__(self, anchors, iou_threshold=0.5,
                 gamma=2.0, alpha=0.25, device="cuda"):
        super().__init__()
        self.anchors       = anchors.to(device)
        self.iou_threshold = iou_threshold
        self.gamma         = gamma
        self.alpha         = alpha
        self.device        = device

        # Mild per-class weighting — focal loss does most of the work,
        # this just nudges the historically-hard small items up a bit
        # and the over-represented helmet class down. Background gets
        # its own α via focal_loss.
        weights = torch.ones(NUM_CLASSES)
        weights[BACKGROUND_IDX]            = 1.0
        weights[CLASS_TO_IDX["helmet"]]    = 0.8
        weights[CLASS_TO_IDX["no_helmet"]] = 1.0
        weights[CLASS_TO_IDX["vest"]]      = 1.0
        weights[CLASS_TO_IDX["no_vest"]]   = 1.0
        weights[CLASS_TO_IDX["gloves"]]    = 1.3
        weights[CLASS_TO_IDX["boots"]]     = 1.3
        weights[CLASS_TO_IDX["mask"]]      = 1.5
        weights[CLASS_TO_IDX["no_mask"]]   = 1.5
        self.register_buffer("class_weights", weights)

    def focal_loss(self, logits, targets):
        """Per-anchor focal cross-entropy. `targets` may include -1
        for ignore (anchors with ambiguous IoU)."""
        valid = targets >= 0
        if not valid.any():
            return torch.zeros((), device=logits.device)
        logits_v  = logits[valid]
        targets_v = targets[valid]

        ce = torch.nn.functional.cross_entropy(
            logits_v, targets_v,
            weight=self.class_weights.to(logits.device),
            reduction="none",
        )
        pt    = torch.exp(-ce)
        # α weights foreground (positives) vs background
        alpha = torch.where(
            targets_v > BACKGROUND_IDX,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha),
        )
        loss = alpha * ((1 - pt) ** self.gamma) * ce

        out = torch.zeros_like(targets, dtype=loss.dtype)
        out[valid] = loss
        return out

    def forward(self, cls_logits, box_preds, targets):
        B = cls_logits.size(0)
        N = self.anchors.size(0)

        gt_cls  = torch.zeros(B, N, dtype=torch.long).to(self.device)
        gt_locs = torch.zeros(B, N, 4).to(self.device)

        for i, target in enumerate(targets):
            gt_boxes  = target["boxes"].to(self.device)
            gt_labels = target["labels"].to(self.device)

            if gt_boxes.shape[0] == 0:
                continue

            iou = self._iou_cx(self.anchors, gt_boxes)
            best_gt_iou,  best_gt_idx  = iou.max(dim=1)
            _,            best_anc_idx = iou.max(dim=0)

            # Force best-matching anchor per GT to be positive
            best_gt_idx[best_anc_idx] = torch.arange(
                gt_boxes.size(0)).to(self.device)
            best_gt_iou[best_anc_idx] = 1.0

            matched_labels = gt_labels[best_gt_idx]
            # Below threshold but above 0.4 → ignore; below 0.4 → background
            ignore_mask = (best_gt_iou < self.iou_threshold) & \
                          (best_gt_iou >= 0.4)
            bg_mask     = best_gt_iou < 0.4
            matched_labels[ignore_mask] = -1
            matched_labels[bg_mask]     = 0

            gt_cls[i]  = matched_labels
            gt_locs[i] = self._encode(gt_boxes[best_gt_idx],
                                      self.anchors)

        pos_mask  = gt_cls > BACKGROUND_IDX
        num_pos   = pos_mask.sum().clamp(min=1).float()

        # Focal classification loss — averaged over positives, as in
        # RetinaNet. Ignored anchors contribute 0.
        cls_loss_per_anchor = self.focal_loss(
            cls_logits.view(-1, cls_logits.size(-1)),
            gt_cls.view(-1),
        ).view(B, N)
        cls_loss = cls_loss_per_anchor.sum() / num_pos

        # Localisation loss — smooth L1 over positive anchors only
        if pos_mask.any():
            pos_idx  = pos_mask.unsqueeze(-1).expand_as(box_preds)
            loc_loss = torch.nn.functional.smooth_l1_loss(
                box_preds[pos_idx].view(-1, 4),
                gt_locs[pos_mask].view(-1, 4),
                reduction="sum",
            ) / num_pos
        else:
            loc_loss = torch.zeros((), device=self.device)

        return cls_loss + loc_loss, cls_loss, loc_loss

    def _iou_cx(self, a, b):
        return self._box_iou(self._cx_to_corner(a),
                             self._cx_to_corner(b))

    def _cx_to_corner(self, boxes):
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=1)

    def _box_iou(self, a, b):
        N, M = a.size(0), b.size(0)
        lt   = torch.max(a[:,:2].unsqueeze(1).expand(N,M,2),
                         b[:,:2].unsqueeze(0).expand(N,M,2))
        rb   = torch.min(a[:,2:].unsqueeze(1).expand(N,M,2),
                         b[:,2:].unsqueeze(0).expand(N,M,2))
        wh   = (rb - lt).clamp(min=0)
        inter = wh[:,:,0] * wh[:,:,1]
        area_a = ((a[:,2]-a[:,0])*(a[:,3]-a[:,1])).unsqueeze(1)
        area_b = ((b[:,2]-b[:,0])*(b[:,3]-b[:,1])).unsqueeze(0)
        return inter / (area_a + area_b - inter + 1e-6)

    def _encode(self, gt, anchors, variances=(0.1, 0.2)):
        g_cx = (gt[:,0]-anchors[:,0]) / (variances[0]*anchors[:,2])
        g_cy = (gt[:,1]-anchors[:,1]) / (variances[0]*anchors[:,3])
        g_w  = torch.log(gt[:,2]/anchors[:,2]+1e-6) / variances[1]
        g_h  = torch.log(gt[:,3]/anchors[:,3]+1e-6) / variances[1]
        return torch.stack([g_cx, g_cy, g_w, g_h], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model      = PPEDetector(num_classes=NUM_CLASSES).to(device)
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters()
                     if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Num classes          : {NUM_CLASSES}")

    dummy = torch.randn(2, 3, 300, 300).to(device)
    cls_logits, box_preds = model(dummy)
    print(f"cls_logits shape : {cls_logits.shape}")
    print(f"box_preds shape  : {box_preds.shape}")

    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    print(f"Total anchors    : {anchors.shape[0]}")

    targets = [
        {"boxes" : torch.tensor([[0.5, 0.5, 0.3, 0.4]]),
         "labels": torch.tensor([CLASS_TO_IDX["helmet"]])},
        {"boxes" : torch.tensor([[0.3, 0.4, 0.2, 0.3]]),
         "labels": torch.tensor([CLASS_TO_IDX["vest"]])},
    ]
    criterion = SSDLoss(anchors, device=str(device))
    loss, cls_l, loc_l = criterion(cls_logits, box_preds, targets)
    print(f"Total loss : {loss.item():.4f}")
    print(f"Cls loss   : {cls_l.item():.4f}")
    print(f"Loc loss   : {loc_l.item():.4f}")
    print("\nPPE model architecture OK")
