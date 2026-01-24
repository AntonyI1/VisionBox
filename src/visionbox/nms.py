"""Non-Maximum Suppression for object detection."""

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center format [x, y, w, h] to corner format [x1, y1, x2, y2]."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: (N, 4) boxes in xyxy format
        box2: (M, 4) boxes in xyxy format

    Returns:
        (N, M) IoU matrix
    """
    b1 = box1.unsqueeze(1)  # (N, 1, 4)
    b2 = box2.unsqueeze(0)  # (1, M, 4)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    return inter / (area1 + area2 - inter + 1e-6)


def nms(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> list[torch.Tensor]:
    """
    Apply Non-Maximum Suppression to YOLO predictions.

    Args:
        predictions: Raw model output (batch, num_preds, 85)
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for suppression

    Returns:
        List of detection tensors per image, each (N, 6): [x1, y1, x2, y2, conf, class_id]
    """
    batch_size = predictions.shape[0]
    results = []

    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]

        # Filter by objectness
        mask = pred[:, 4] > conf_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            results.append(torch.zeros((0, 6), device=predictions.device))
            continue

        # Get class scores and best class
        class_scores = pred[:, 5:] * pred[:, 4:5]
        class_conf, class_id = class_scores.max(dim=1)

        # Filter by class confidence
        mask = class_conf > conf_threshold
        pred = pred[mask]
        class_conf = class_conf[mask]
        class_id = class_id[mask]

        if pred.shape[0] == 0:
            results.append(torch.zeros((0, 6), device=predictions.device))
            continue

        boxes = xywh_to_xyxy(pred[:, :4])

        # NMS per class
        keep_indices = []
        for cls in class_id.unique():
            cls_mask = class_id == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = class_conf[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            sorted_idx = cls_scores.argsort(descending=True)
            cls_boxes = cls_boxes[sorted_idx]
            cls_indices = cls_indices[sorted_idx]

            keep = []
            while len(cls_boxes) > 0:
                keep.append(cls_indices[0].item())
                if len(cls_boxes) == 1:
                    break

                ious = compute_iou(cls_boxes[0:1], cls_boxes[1:]).squeeze(0)
                mask = ious < iou_threshold
                cls_boxes = cls_boxes[1:][mask]
                cls_indices = cls_indices[1:][mask]

            keep_indices.extend(keep)

        if not keep_indices:
            results.append(torch.zeros((0, 6), device=predictions.device))
            continue

        keep_indices = torch.tensor(keep_indices, device=predictions.device)
        detections = torch.cat([
            boxes[keep_indices],
            class_conf[keep_indices].unsqueeze(1),
            class_id[keep_indices].unsqueeze(1).float()
        ], dim=1)

        results.append(detections)

    return results
