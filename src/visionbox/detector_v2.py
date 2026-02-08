"""Multi-model YOLO detector with auto backend selection (TensorRT > OpenVINO > PyTorch)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class ModelConfig:
    path: str
    class_offset: int = 0
    class_names: dict = None
    conf_threshold: float = 0.25
    class_conf: dict = None


class MultiModelDetector:
    def __init__(
        self,
        model_configs: list[ModelConfig] = None,
        device: str = 'auto',
        imgsz: int = 640
    ):
        self.device = self._resolve_device(device)
        self.imgsz = imgsz

        if model_configs is None:
            model_configs = [ModelConfig('yolov8n.pt')]

        self.models: list[tuple[YOLO, ModelConfig]] = []
        self.class_names: dict[int, str] = {}

        for config in model_configs:
            model_path = self._find_best_model(config.path)
            model = YOLO(model_path, task='detect')

            if model_path.endswith('.pt') and self.device != 'cpu':
                model.to(self.device)

            self.models.append((model, config))

            for orig_id, name in model.names.items():
                unified_id = orig_id + config.class_offset
                if config.class_names and orig_id in config.class_names:
                    self.class_names[unified_id] = config.class_names[orig_id]
                else:
                    self.class_names[unified_id] = name

            print(f"  Loaded: {Path(model_path).name} ({len(model.names)} classes)")

        print(f"Loaded {len(self.models)} model(s) on {self.device}")
        print(f"Total classes: {len(self.class_names)}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == 'auto':
            return 'cuda' if _has_cuda() else 'cpu'
        return device

    @staticmethod
    def _find_best_model(path: str) -> str:
        p = Path(path)
        engine = p.with_suffix('.engine')
        if engine.exists():
            print(f"  Using TensorRT: {engine.name}")
            return str(engine)
        openvino_dir = p.with_name(p.stem + '_openvino_model')
        if openvino_dir.is_dir():
            print(f"  Using OpenVINO: {openvino_dir.name}")
            return str(openvino_dir)
        models_openvino = Path('models') / (p.stem + '_openvino_model')
        if models_openvino.is_dir():
            print(f"  Using OpenVINO: {models_openvino}")
            return str(models_openvino)
        return path

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] = None
    ) -> list[dict]:
        all_detections = []

        for model, config in self.models:
            conf = config.conf_threshold if config.conf_threshold else conf_threshold
            use_half = self.device not in ('cpu', 'auto')
            results = model(frame, conf=conf, iou=iou_threshold, verbose=False,
                            half=use_half, imgsz=self.imgsz)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
                clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

                for i in range(len(boxes)):
                    box = xyxy[i]
                    conf_score = float(confs[i])
                    orig_class_id = int(clss[i])
                    unified_class_id = orig_class_id + config.class_offset

                    if classes is not None and unified_class_id not in classes:
                        continue

                    if config.class_conf:
                        min_conf = config.class_conf.get(orig_class_id, conf)
                        if conf_score < min_conf:
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
        """Returns (N, 6) array: [x1, y1, x2, y2, confidence, class_id]."""
        detections = self.detect(frame, conf_threshold, iou_threshold, classes)
        if not detections:
            return np.empty((0, 6))
        return np.array([[*d['box'], d['confidence'], d['class_id']] for d in detections])


def export_tensorrt(model_name: str = 'yolov8s.pt', imgsz: int = 1280):
    model = YOLO(model_name)
    model.export(format='engine', half=True, imgsz=imgsz)
    print(f"Exported {model_name} → TensorRT FP16 engine (imgsz={imgsz})")


def export_openvino(model_name: str = 'yolov8n.pt', imgsz: int = 640):
    model = YOLO(model_name)
    model.export(format='openvino', imgsz=imgsz, half=False)
    print(f"Exported {model_name} → OpenVINO IR (imgsz={imgsz})")


def create_surveillance_detector(device: str = 'auto') -> MultiModelDetector:
    """Create multi-model detector: COCO + license plate + bottle (if available)."""
    models_dir = Path(__file__).parent.parent.parent / 'models'

    configs = [ModelConfig(path='yolov8n.pt', class_offset=0, conf_threshold=0.25)]

    lp_model = models_dir / 'license-plate-finetune-v1n.pt'
    if lp_model.exists():
        configs.append(ModelConfig(
            path=str(lp_model), class_offset=80,
            class_names={0: 'license_plate'}, conf_threshold=0.3
        ))

    bottle_model = models_dir / 'bottle-custom.pt'
    if bottle_model.exists():
        configs.append(ModelConfig(
            path=str(bottle_model), class_offset=81,
            class_names={0: 'bottle'}, conf_threshold=0.3
        ))

    return MultiModelDetector(configs, device=device)


CLASS_PRESETS_V2 = {
    'outdoor': [0, 1, 2, 3, 5, 7, 14, 15, 16, 80],
    'indoor': [0, 39, 41, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67, 73, 74, 81],
    'vehicles': [1, 2, 3, 5, 7, 80],
    'all': None,
}
