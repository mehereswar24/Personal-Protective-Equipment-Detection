import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from step2_model import MobileNetV2SSD, AnchorGenerator
import torchvision.transforms as T

# ── Config ──────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH     = "models/person_detector_best.pth"
CONF_THRESHOLD = 0.7      # lower = more detections, higher = fewer
NMS_THRESHOLD  = 0.4       # overlap threshold for NMS
INPUT_SIZE     = 300
# ────────────────────────────────────────────────────────

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def decode_boxes(box_preds, anchors, variances=(0.1, 0.2)):
    """Decode SSD offsets back to cx,cy,w,h in 0-1 range."""
    cx = box_preds[:, 0] * variances[0] * anchors[:, 2] + anchors[:, 0]
    cy = box_preds[:, 1] * variances[0] * anchors[:, 3] + anchors[:, 1]
    w  = torch.exp(box_preds[:, 2] * variances[1]) * anchors[:, 2]
    h  = torch.exp(box_preds[:, 3] * variances[1]) * anchors[:, 3]
    # convert to x1,y1,x2,y2
    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=1)


def nms(boxes, scores, iou_threshold=0.4):
    """Pure PyTorch NMS."""
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ix1  = x1[rest].clamp(min=x1[i].item())
        iy1  = y1[rest].clamp(min=y1[i].item())
        ix2  = x2[rest].clamp(max=x2[i].item())
        iy2  = y2[rest].clamp(max=y2[i].item())

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def load_model():
    model = MobileNetV2SSD(num_classes=2).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from epoch {ckpt['epoch']} "
          f"(val_loss: {ckpt['val_loss']:.4f})")
    return model


@torch.no_grad()
def detect(model, anchors, frame_rgb):
    """
    Run person detection on a single RGB frame.
    Returns list of (x1, y1, x2, y2, score) in pixel coordinates.
    """
    h, w = frame_rgb.shape[:2]

    # preprocess
    img = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # forward
    cls_logits, box_preds = model(tensor)
    cls_logits = cls_logits[0]   # [N, 2]
    box_preds  = box_preds[0]    # [N, 4]

    # softmax scores for person class (class 1)
    scores = torch.softmax(cls_logits, dim=1)[:, 1]

    # decode boxes
    boxes = decode_boxes(box_preds, anchors.to(DEVICE))

    # filter by confidence
    mask   = scores > CONF_THRESHOLD
    boxes  = boxes[mask]
    scores = scores[mask]

    if boxes.shape[0] == 0:
        return []

    # NMS
    keep   = nms(boxes, scores, NMS_THRESHOLD)
    boxes  = boxes[keep]
    scores = scores[keep]

    # convert to pixel coordinates
    results = []
    for box, score in zip(boxes, scores):
        x1 = int(box[0].item() * w)
        y1 = int(box[1].item() * h)
        x2 = int(box[2].item() * w)
        y2 = int(box[3].item() * h)
        results.append((x1, y1, x2, y2, score.item()))

    return results


def draw_detections(frame, detections):
    """Draw bounding boxes on frame."""
    for (x1, y1, x2, y2, score) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def test_on_image(image_path):
    """Test detector on a single image."""
    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    model      = load_model()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return

    frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections  = detect(model, anchors, frame_rgb)

    print(f"Detected {len(detections)} person(s):")
    for i, (x1, y1, x2, y2, score) in enumerate(detections):
        print(f"  Person {i+1}: box=({x1},{y1},{x2},{y2}) score={score:.3f}")

    frame = draw_detections(frame, detections)

    out_path = "output/person_detection_test.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Result saved to: {out_path}")

    # show result
    print(f"Result saved to: {out_path}")


def test_on_video(video_path):
    """Test detector on a video file or webcam (0 for webcam)."""
    anchor_gen = AnchorGenerator()
    anchors    = anchor_gen()
    model      = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    # video writer
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(
        "output/person_detection_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (fw, fh)
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect(model, anchors, frame_rgb)
        frame      = draw_detections(frame, detections)

        # fps counter
        frame_count += 1
        cv2.putText(frame, f"Persons: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        out.write(frame)
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Output saved to: output/person_detection_output.mp4")


if __name__ == "__main__":
    # test on an image from the INRIA dataset
    test_image = "INRIAPerson/Test/JPEGImages/crop_000001.png"

    if not os.path.exists(test_image):
        # pick any image from the test set
        import glob
        images = glob.glob("INRIAPerson/Test/JPEGImages/*.jpg")
        if images:
            test_image = images[0]

    print(f"Testing on: {test_image}")
    test_on_image(test_image)