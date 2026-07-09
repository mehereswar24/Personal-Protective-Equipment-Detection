import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# 6 anchors per cell: 2 scales x 3 aspect ratios. Must stay in sync with
# AnchorGenerator.
NUM_ANCHORS = 6


class MobileNetV2SSD(nn.Module):
    """
    SSD-style person detector built on MobileNetV2 backbone.

    Architecture:
        MobileNetV2 (pretrained ImageNet) → feature extractor
        Extra conv layers                 → multi-scale feature maps
        Detection heads                   → class scores + box offsets

    Output (during training):
        cls_logits : list of [B, num_anchors * num_classes, H, W]
        box_preds  : list of [B, num_anchors * 4, H, W]

    Anchor scheme (per feature map):
        3 aspect ratios × 1 scale = 3 anchors per cell
        Feature maps   : 19×19, 10×10, 5×5, 3×3, 1×1
        Total anchors  : 19²×3 + 10²×3 + 5²×3 + 3²×3 + 1²×3 = 1·regrets
                       = 1083 + 300 + 75 + 27 + 3 = 1488
    """

    def __init__(self, num_classes=2):  # 0=background, 1=person
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone: MobileNetV2 ──────────────────────────────────
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Feature map 1: after layer 14 → stride 16 → 19×19 for 300px input
        self.feature1 = nn.Sequential(*backbone.features[:14])   # out: 96 ch

        # Feature map 2: rest of MobileNetV2 → 10×10
        self.feature2 = nn.Sequential(*backbone.features[14:])   # out: 1280 ch

        # ── Extra layers for smaller feature maps ─────────────────
        self.extra1 = self._extra_block(1280, 256, stride=2)     # 5×5
        self.extra2 = self._extra_block(256,  128, stride=2)     # 3×3
        self.extra3 = self._extra_block(128,  128, stride=3)     # 1×1

        # ── Detection heads ───────────────────────────────────────
        # (class + box head for each feature map)
        self.num_anchors = NUM_ANCHORS   # 2 scales x 3 aspect ratios

        feature_channels = [96, 1280, 256, 128, 128]
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(ch, self.num_anchors * num_classes, kernel_size=3, padding=1)
            for ch in feature_channels
        ])
        self.box_heads = nn.ModuleList([
            nn.Conv2d(ch, self.num_anchors * 4, kernel_size=3, padding=1)
            for ch in feature_channels
        ])

        # ── Init extra layers + heads ─────────────────────────────
        self._init_weights()

    def _extra_block(self, in_ch, out_ch, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, out_ch // 2), num_channels=out_ch // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3,
                      stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.ReLU6(inplace=True),
        )

    def _init_weights(self):
        for m in self.extra1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        for m in self.extra2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        for m in self.extra3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        for head in self.cls_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0)
        for head in self.box_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        # ── Extract feature maps ──────────────────────────────────
        f1 = self.feature1(x)          # [B, 96,   19, 19]
        f2 = self.feature2(f1)         # [B, 1280, 10, 10]
        f3 = self.extra1(f2)           # [B, 256,   5,  5]
        f4 = self.extra2(f3)           # [B, 128,   3,  3]
        f5 = self.extra3(f4)           # [B, 128,   1,  1]

        features = [f1, f2, f3, f4, f5]

        # ── Apply detection heads ─────────────────────────────────
        cls_logits = []
        box_preds  = []

        for feat, cls_head, box_head in zip(
                features, self.cls_heads, self.box_heads):

            cls = cls_head(feat)   # [B, num_anchors*num_classes, H, W]
            box = box_head(feat)   # [B, num_anchors*4, H, W]

            # reshape → [B, H*W*num_anchors, num_classes or 4]
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(B, -1, self.num_classes)

            box = box.permute(0, 2, 3, 1).contiguous()
            box = box.view(B, -1, 4)

            cls_logits.append(cls)
            box_preds.append(box)

        # concat across all feature maps → [B, total_anchors, ...]
        cls_logits = torch.cat(cls_logits, dim=1)
        box_preds  = torch.cat(box_preds,  dim=1)

        return cls_logits, box_preds


class AnchorGenerator(nn.Module):
    """
    SSD anchors with 2 scales per feature map x 3 aspect ratios = 6 anchors
    per cell. Covers small (distant), medium and large (close-up) persons.
    """

    def __init__(self):
        super().__init__()
        # (feature_map_size, primary_scale)
        # Smallest scale 0.06 covers ~18px persons in a 300px input —
        # critical for the distant/aerial scenes the old model missed.
        levels = [
            (19, 0.06),
            (10, 0.18),
            ( 5, 0.34),
            ( 3, 0.54),
            ( 1, 0.75),
        ]
        next_scales = [s for (_, s) in levels[1:]] + [0.95]
        # Wider aspect ratios — persons are tall (h > w), so emphasise that.
        ratios = [1.0, 0.5, 2.0]

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
    RetinaNet-style focal loss for SSD. Focal loss handles the extreme
    pos/neg imbalance on its own, so we do NOT stack hard-negative mining
    on top. Per-positive normalisation matches the RetinaNet formulation.

    IoU >= iou_threshold (0.5)   -> positive (person)
    0.4 <= IoU < 0.5             -> ignore (don't contribute to loss)
    IoU < 0.4                    -> background
    """

    def __init__(self, anchors, iou_threshold=0.5,
                 gamma=2.0, alpha=0.25, device="cuda"):
        super().__init__()
        self.anchors       = anchors.to(device)
        self.iou_threshold = iou_threshold
        self.gamma         = gamma
        self.alpha         = alpha
        self.device        = device

    def focal_loss(self, logits, targets):
        valid = targets >= 0
        if not valid.any():
            return torch.zeros((), device=logits.device)
        logits_v  = logits[valid]
        targets_v = targets[valid]
        ce = F.cross_entropy(logits_v, targets_v, reduction="none")
        pt = torch.exp(-ce)
        alpha = torch.where(
            targets_v > 0,
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

            # Force best-matching anchor per GT to be a positive
            best_gt_idx[best_anc_idx] = torch.arange(
                gt_boxes.size(0)).to(self.device)
            best_gt_iou[best_anc_idx] = 1.0

            matched_labels = gt_labels[best_gt_idx]
            ignore_mask = (best_gt_iou < self.iou_threshold) & \
                          (best_gt_iou >= 0.4)
            bg_mask     = best_gt_iou < 0.4
            matched_labels[ignore_mask] = -1
            matched_labels[bg_mask]     = 0

            gt_cls[i]  = matched_labels
            gt_locs[i] = self._encode(gt_boxes[best_gt_idx], self.anchors)

        pos_mask = gt_cls > 0
        num_pos  = pos_mask.sum().clamp(min=1).float()

        cls_loss_per_anchor = self.focal_loss(
            cls_logits.view(-1, cls_logits.size(-1)),
            gt_cls.view(-1),
        ).view(B, N)
        cls_loss = cls_loss_per_anchor.sum() / num_pos

        if pos_mask.any():
            pos_idx  = pos_mask.unsqueeze(-1).expand_as(box_preds)
            loc_loss = F.smooth_l1_loss(
                box_preds[pos_idx].view(-1, 4),
                gt_locs[pos_mask].view(-1, 4),
                reduction="sum",
            ) / num_pos
        else:
            loc_loss = torch.zeros((), device=self.device)

        return cls_loss + loc_loss, cls_loss, loc_loss

    def _iou_cx(self, a, b):
        """IoU between anchors a [N,4] and gt b [M,4], both cx,cy,w,h."""
        a = self._cx_to_corner(a)
        b = self._cx_to_corner(b)
        return self._box_iou(a, b)

    def _cx_to_corner(self, boxes):
        return torch.cat([
            boxes[:, :2] - boxes[:, 2:] / 2,
            boxes[:, :2] + boxes[:, 2:] / 2
        ], dim=1)

    def _box_iou(self, a, b):
        """a: [N,4], b: [M,4] corner format → [N,M] IoU."""
        N, M = a.size(0), b.size(0)
        lt = torch.max(a[:, :2].unsqueeze(1).expand(N, M, 2),
                       b[:, :2].unsqueeze(0).expand(N, M, 2))
        rb = torch.min(a[:, 2:].unsqueeze(1).expand(N, M, 2),
                       b[:, 2:].unsqueeze(0).expand(N, M, 2))
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = ((a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])).unsqueeze(1)
        area_b = ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])).unsqueeze(0)
        return inter / (area_a + area_b - inter + 1e-6)

    def _encode(self, gt, anchors, variances=(0.1, 0.2)):
        """Encode gt boxes as offsets relative to anchors."""
        g_cx = (gt[:, 0] - anchors[:, 0]) / (variances[0] * anchors[:, 2])
        g_cy = (gt[:, 1] - anchors[:, 1]) / (variances[0] * anchors[:, 3])
        g_w  = torch.log(gt[:, 2] / anchors[:, 2] + 1e-6) / variances[1]
        g_h  = torch.log(gt[:, 3] / anchors[:, 3] + 1e-6) / variances[1]
        return torch.stack([g_cx, g_cy, g_w, g_h], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # build model
    model = MobileNetV2SSD(num_classes=2).to(device)

    # count parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    # test forward pass
    dummy = torch.randn(2, 3, 300, 300).to(device)
    cls_logits, box_preds = model(dummy)
    print(f"cls_logits shape : {cls_logits.shape}")   # [2, 1488, 2]
    print(f"box_preds shape  : {box_preds.shape}")    # [2, 1488, 4]

    # test anchor generator
    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    print(f"Total anchors    : {anchors.shape[0]}")   # 1488

    # test loss
    targets = [
        {"boxes" : torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
         "labels": torch.tensor([1])},
        {"boxes" : torch.tensor([[0.3, 0.4, 0.15, 0.35]]),
         "labels": torch.tensor([1])},
    ]
    criterion = SSDLoss(anchors, device=str(device))
    loss, cls_loss, loc_loss = criterion(cls_logits, box_preds, targets)
    print(f"Total loss : {loss.item():.4f}")
    print(f"Cls loss   : {cls_loss.item():.4f}")
    print(f"Loc loss   : {loc_loss.item():.4f}")
    print("\nModel architecture OK")