"""
Detector construction — torchvision, COCO-pretrained, no YOLO.

Replaces the hand-rolled MobileNetV2-SSD, which carried a set of bugs that were
individually fixable but collectively not worth keeping:

  * focal-loss classification head bias never initialised to the prior, so
    epoch 1 opened at loss 74
  * α=0.25 applied in a *softmax* setting (its justification is sigmoid/
    one-vs-all), producing a ~3x anti-foreground tilt
  * class weights multiplied into the CE *before* `pt = exp(-ce)`, so the focal
    modulator no longer measured "how easy is this example"
  * anchors never clipped to the image; forced best-anchor positives at IoU≈0
    generating extreme regression targets
  * smallest anchor scale 0.10 vs median glove size 0.069 → most gloves could
    not reach the 0.5 IoU match threshold *at all*

FCOS is anchor-free, which deletes that entire bug class rather than patching
it, and its P3 (stride-8) level at 800px gives roughly 0.04 relative resolution
— below the glove scale, so gloves become learnable for the first time.

`person` detection uses a COCO-pretrained Faster R-CNN directly: COCO's person
class is far stronger than anything ~8k leakage-prone images can produce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights, fcos_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2,
)

from .taxonomy import CLASSES

# torchvision detectors reserve 0 for background
NUM_CLASSES_WITH_BG = len(CLASSES) + 1


def build_ppe_model(arch: str = "fcos", num_classes: int = NUM_CLASSES_WITH_BG,
                    pretrained: bool = True, min_size: int = 800,
                    max_size: int = 1333, trainable_backbone_layers: int = 3):
    """PPE detector. `arch` is 'fcos' (default) or 'retinanet' (A/B second arm)."""
    if arch == "fcos":
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
        )
        _replace_fcos_head(model, num_classes)
    elif arch == "retinanet":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
        )
        _replace_retinanet_head(model, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r} (expected 'fcos' or 'retinanet')")
    return model


def _prior_bias(prob: float = 0.01) -> float:
    """RetinaNet/FCOS prior bias: -log((1-π)/π).

    Omitting this is why the old model's first epoch exploded — with a zero
    bias every one of ~3000 locations starts out confidently foreground.
    """
    import math
    return -math.log((1 - prob) / prob)


def _replace_fcos_head(model, num_classes: int) -> None:
    head = model.head.classification_head
    in_channels = head.conv[0][0].in_channels if hasattr(head.conv[0], "__getitem__") \
        else head.conv[0].in_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    head.num_classes = num_classes
    head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
                                kernel_size=3, stride=1, padding=1)
    nn.init.normal_(head.cls_logits.weight, std=0.01)
    nn.init.constant_(head.cls_logits.bias, _prior_bias())


def _replace_retinanet_head(model, num_classes: int) -> None:
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes,
        norm_layer=lambda c: nn.GroupNorm(32, c),
    )
    # torchvision already applies the prior bias here; assert rather than assume
    bias = model.head.classification_head.cls_logits.bias
    assert float(bias.mean()) < -2.0, "prior bias not applied to RetinaNet head"


def build_person_model(pretrained: bool = True, min_size: int = 640,
                       max_size: int = 1333):
    """COCO-pretrained person detector, used as-is (person == COCO class 1).

    Deliberately NOT fine-tuned initially: the project's person data is ~8k
    leakage-prone images, and COCO person detection is stronger than what that
    can produce. Revisit only if measured recall on real footage is poor.
    """
    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None,
        min_size=min_size, max_size=max_size,
    )


COCO_PERSON_LABEL = 1


class MaskedLossWrapper(nn.Module):
    """Applies per-image class-presence masking to the classification loss.

    torchvision's detection heads compute a single scalar classification loss
    internally, so a per-class mask cannot be injected without reaching inside.
    Rather than fork the head, we take the honest, simple route: split the batch
    by its class-mask signature and run one forward per distinct signature, with
    the un-annotated classes' ground truth removed. Images that cannot testify
    about a class contribute no gradient that would push that class toward
    background.

    Batches are usually 1-2 signatures deep (a batch tends to come from a few
    datasets), so the cost is small compared to getting the supervision right.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        if targets is None or not self.training:
            return self.model(images)

        groups: dict[tuple, list[int]] = {}
        for i, t in enumerate(targets):
            key = tuple(t.get("class_mask", torch.ones(len(CLASSES))).tolist())
            groups.setdefault(key, []).append(i)

        total: dict[str, torch.Tensor] = {}
        for key, idxs in groups.items():
            sub_images = [images[i] for i in idxs]
            sub_targets = []
            for i in idxs:
                t = targets[i]
                # +1: the ONLY place our 0-based CLASS_TO_IDX meets torchvision's
                # convention, where 0 is background and real classes start at 1.
                # Passing our indices through unshifted trains `person` (index 0)
                # as background so it can never be detected, and shifts every
                # other class one slot against the `- 1` decode in run_eval.
                # The symptom is a healthy falling loss with mAP near zero, and
                # AP of exactly 0.0 for the classes at both ends of the shift.
                # Everything outside this line stays 0-based.
                sub_targets.append({"boxes": t["boxes"],
                                    "labels": t["labels"] + 1})
            losses = self.model(sub_images, sub_targets)
            weight = len(idxs) / len(targets)
            for k, v in losses.items():
                total[k] = total.get(k, 0.0) + v * weight
        return total


def freeze_backbone_bn(model: nn.Module) -> None:
    """Freeze BatchNorm running stats.

    At batch sizes of 4-8 (all an 8GB laptop GPU allows at 800px) BN statistics
    are too noisy to be useful, and the old code's "frozen" backbone still
    updated them because it only set requires_grad=False.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


class ModelEMA:
    """Exponential moving average of weights — reliably worth 1-2 AP."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])
