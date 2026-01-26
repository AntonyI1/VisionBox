"""
Multi-model detector using YOLOv8/v11 via Ultralytics.

Supports loading multiple models and merging their detections.
Each model can detect different classes (e.g., COCO + license plates).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class ModelConfig:
    """Configuration for a detection model."""
    path: str                    # Model path or name (e.g., 'yolov8n.pt')
    class_offset: int = 0        # Offset to add to class IDs (for multi-model merging)
    class_names: dict = None     # Override class names {id: name}
    conf_threshold: float = 0.25


class MultiModelDetector:
    """
    Detector that runs multiple YOLO models and merges results.

    Example:
        detector = MultiModelDetector([
            ModelConfig('yolov8n.pt'),  # COCO classes 0-79
            ModelConfig('models/license-plate.pt', class_offset=80),  # License plate = 80
        ])
        detections = detector.detect(frame)
    """

    # Standard COCO class names (80 classes)
    COCO_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        model_configs: list[ModelConfig] = None,
        device: str = 'cuda'
    ):
        """
        Initialize multi-model detector.

        Args:
            model_configs: List of ModelConfig for each model to load.
                          If None, loads default YOLOv8n.
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'

        if model_configs is None:
            model_configs = [ModelConfig('yolov8n.pt')]

        self.models: list[tuple[YOLO, ModelConfig]] = []
        self.class_names: dict[int, str] = {}

        for config in model_configs:
            model = YOLO(config.path)
            model.to(self.device)
            self.models.append((model, config))

            # Build unified class name mapping
            for orig_id, name in model.names.items():
                unified_id = orig_id + config.class_offset
                if config.class_names and orig_id in config.class_names:
                    self.class_names[unified_id] = config.class_names[orig_id]
                else:
                    self.class_names[unified_id] = name

        print(f"Loaded {len(self.models)} model(s) on {self.device}")
        print(f"Total classes: {len(self.class_names)}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] = None
    ) -> list[dict]:
        """
        Run detection on a frame using all loaded models.

        Args:
            frame: BGR image (numpy array)
            conf_threshold: Minimum confidence (can be overridden per-model)
            iou_threshold: NMS IoU threshold
            classes: Filter to specific class IDs (unified IDs)

        Returns:
            List of detections, each with:
            - box: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int (unified across all models)
            - class_name: str
        """
        all_detections = []

        for model, config in self.models:
            # Use model-specific threshold if set, otherwise use global
            conf = config.conf_threshold if config.conf_threshold else conf_threshold

            # Run inference (ultralytics handles preprocessing)
            results = model(frame, conf=conf, iou=iou_threshold, verbose=False)

            # Extract detections
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf_score = float(boxes.conf[i].cpu())
                    orig_class_id = int(boxes.cls[i].cpu())

                    # Apply class offset for unified IDs
                    unified_class_id = orig_class_id + config.class_offset

                    # Filter by class if specified
                    if classes is not None and unified_class_id not in classes:
                        continue

                    all_detections.append({
                        'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'confidence': conf_score,
                        'class_id': unified_class_id,
                        'class_name': self.class_names.get(unified_class_id, f'class_{unified_class_id}')
                    })

        return all_detections

    def detect_array(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] = None
    ) -> np.ndarray:
        """
        Run detection and return as numpy array for tracker.

        Returns:
            Array of shape (N, 6): [x1, y1, x2, y2, confidence, class_id]
        """
        detections = self.detect(frame, conf_threshold, iou_threshold, classes)
        if not detections:
            return np.empty((0, 6))

        return np.array([
            [*d['box'], d['confidence'], d['class_id']]
            for d in detections
        ])


def create_surveillance_detector(device: str = 'cuda', use_custom_bottle: bool = True) -> MultiModelDetector:
    """
    Create a detector optimized for surveillance use cases.

    Loads:
    - YOLOv8n for COCO classes (person, car, dog, etc.)
    - License plate model
    - Custom bottle model (fine-tuned for insulated bottles/tumblers)

    Class ID mapping:
    - 0-79: COCO classes
    - 80: License plate
    - 81: Bottle (custom fine-tuned)
    """
    models_dir = Path(__file__).parent.parent.parent / 'models'

    configs = [
        # Main COCO model
        ModelConfig(
            path='yolov8n.pt',
            class_offset=0,
            conf_threshold=0.25
        ),
    ]

    # Add license plate model if available
    lp_model = models_dir / 'license-plate-finetune-v1n.pt'
    if lp_model.exists():
        configs.append(ModelConfig(
            path=str(lp_model),
            class_offset=80,  # License plate = class 80
            class_names={0: 'license_plate'},
            conf_threshold=0.3
        ))
        print(f"License plate model: {lp_model.name}")

    # Add custom bottle model if available
    bottle_model = models_dir / 'bottle-custom.pt'
    if bottle_model.exists() and use_custom_bottle:
        configs.append(ModelConfig(
            path=str(bottle_model),
            class_offset=81,  # Custom bottle = class 81
            class_names={0: 'bottle'},
            conf_threshold=0.3
        ))
        print(f"Custom bottle model: {bottle_model.name}")

    return MultiModelDetector(configs, device=device)


# Convenience class presets for filtering
CLASS_PRESETS_V2 = {
    'outdoor': [
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        14,  # bird
        15,  # cat
        16,  # dog
        24,  # backpack
        26,  # handbag
        28,  # suitcase
        80,  # license_plate
    ],
    'indoor': [
        0,   # person
        39,  # bottle (COCO)
        41,  # cup
        56,  # chair
        57,  # couch
        59,  # bed
        60,  # dining table
        62,  # tv
        63,  # laptop
        64,  # mouse
        65,  # remote
        66,  # keyboard
        67,  # cell phone
        73,  # book
        74,  # clock
        81,  # bottle (custom - Yeti/insulated)
    ],
    'vehicles': [
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        80,  # license_plate
    ],
    'all': None  # No filtering
}
