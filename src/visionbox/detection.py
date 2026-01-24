"""YOLO detection pipeline."""

import torch
import numpy as np

from .preprocessing import preprocess
from .nms import nms


class Detector:
    """YOLO-based object detector."""

    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, model_name: str = 'yolov5s', device: str = 'cuda'):
        """
        Initialize detector.

        Args:
            model_name: YOLOv5 model variant (yolov5n/s/m/l/x)
            device: Target device
        """
        self.device = device
        self.model = torch.hub.load(
            'ultralytics/yolov5', model_name,
            pretrained=True, trust_repo=True
        )
        self.model = self.model.to(device)
        self.model.eval()
        self._raw_model = self.model.model

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> list[dict]:
        """
        Run detection on an image.

        Args:
            image: BGR image from OpenCV
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold

        Returns:
            List of detections, each with keys: box, confidence, class_id, class_name
        """
        original_shape = image.shape[:2]
        tensor, scale, padding = preprocess(image, self.device)

        with torch.no_grad():
            predictions = self._raw_model(tensor)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)

        detections = nms(predictions, conf_threshold, iou_threshold)[0]

        if detections.shape[0] > 0:
            detections[:, :4] = self._scale_boxes(
                detections[:, :4].clone(), scale, padding, original_shape
            )

        return self._format_detections(detections)

    def _scale_boxes(
        self,
        boxes: torch.Tensor,
        scale: float,
        padding: tuple[int, int],
        original_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Scale boxes from letterboxed coordinates to original image."""
        pad_w, pad_h = padding
        h_orig, w_orig = original_shape

        boxes[:, 0] -= pad_w
        boxes[:, 1] -= pad_h
        boxes[:, 2] -= pad_w
        boxes[:, 3] -= pad_h
        boxes /= scale

        boxes[:, 0].clamp_(0, w_orig)
        boxes[:, 1].clamp_(0, h_orig)
        boxes[:, 2].clamp_(0, w_orig)
        boxes[:, 3].clamp_(0, h_orig)

        return boxes

    def _format_detections(self, detections: torch.Tensor) -> list[dict]:
        """Convert tensor detections to list of dicts."""
        results = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
            class_id = int(class_id)
            results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class_id': class_id,
                'class_name': self.COCO_CLASSES[class_id]
            })
        return results
