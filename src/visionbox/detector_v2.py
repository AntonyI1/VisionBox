"""Multi-model YOLO detector with auto backend selection (TensorRT > OpenVINO > PyTorch)."""

from contextlib import nullcontext
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


def _as_numpy(tensor) -> np.ndarray:
    return tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.asarray(tensor)


@dataclass
class ModelConfig:
    path: str
    class_offset: int = 0
    class_names: dict[int, str] | None = None
    conf_threshold: float = 0.25
    class_conf: dict[int, float] | None = None


class MultiModelDetector:
    def __init__(
        self,
        model_configs: list[ModelConfig] | None = None,
        device: str = 'auto',
        imgsz: int = 640
    ):
        self._device_arg = device
        self.device = self._resolve_device(device)
        self.ov_device = self._resolve_ov_device(device)
        self.imgsz = imgsz

        if model_configs is None:
            model_configs = [ModelConfig('yolov8n.pt')]

        self.models, self.class_names = self._build_models(model_configs)

        self._warmup()
        print(f"Loaded {len(self.models)} model(s) on {self.effective_device}")
        print(f"Total classes: {len(self.class_names)}")

    def _build_models(self, model_configs):
        """Load each configured model from disk and build the unified class map."""
        models: list[tuple[YOLO, ModelConfig, bool]] = []
        class_names: dict[int, str] = {}
        for config in model_configs:
            model_path = self._find_best_model(config.path)
            is_openvino = Path(model_path).is_dir()
            model = YOLO(model_path, task='detect')

            if model_path.endswith('.pt') and self.device != 'cpu':
                model.to(self.device)

            models.append((model, config, is_openvino))

            for orig_id, name in model.names.items():
                unified_id = orig_id + config.class_offset
                if config.class_names and orig_id in config.class_names:
                    class_names[unified_id] = config.class_names[orig_id]
                else:
                    class_names[unified_id] = name

            print(f"  Loaded: {Path(model_path).name} ({len(model.names)} classes)")
        return models, class_names

    def reload(self, swap_lock=None) -> dict:
        """Rebuild every model from disk (after an on-box training hot-swap) and re-warm.

        Loading and warming the new models happens outside swap_lock so live inference
        keeps going; only the final pointer swap runs under it. The OpenVINO device is
        re-resolved so a model that fell back to CPU can return to the iGPU.
        """
        self.ov_device = self._resolve_ov_device(self._device_arg)
        configs = [config for _, config, _ in self.models]
        new_models, new_class_names = self._build_models(configs)
        self._warmup(new_models)
        with swap_lock if swap_lock is not None else nullcontext():
            self.models, self.class_names = new_models, new_class_names
        print(f"Reloaded {len(self.models)} model(s) on {self.effective_device}", flush=True)
        return {
            'models': len(self.models),
            'classes': len(self.class_names),
            'device': self.effective_device,
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        # 'gpu' selects the iGPU for OpenVINO (via _resolve_ov_device); for the torch
        # (.pt) path it must map to a real torch device, never the literal 'gpu'.
        if device in ('auto', 'gpu'):
            return 'cuda' if _has_cuda() else 'cpu'
        return device

    @staticmethod
    def _resolve_ov_device(device: str) -> str:
        """OpenVINO target for Ultralytics — prefer the Intel iGPU, fall back to CPU.

        An explicit 'intel:gpu' is dramatically faster than letting Ultralytics
        default to OpenVINO AUTO on this hardware, so we never rely on AUTO.
        """
        if device == 'cpu':
            return 'intel:cpu'
        try:
            from openvino import Core
            available = Core().available_devices
        except Exception:
            available = []
        return 'intel:gpu' if 'GPU' in available else 'intel:cpu'

    @property
    def effective_device(self) -> str:
        if any(is_ov for _, _, is_ov in self.models):
            return self.ov_device
        return self.device

    def _warmup(self, models=None):
        """Prime each model (compiles GPU kernels) so the first real frame isn't slow."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for model, _, is_openvino in (models if models is not None else self.models):
            kw = {'imgsz': self.imgsz, 'verbose': False}
            if is_openvino:
                kw['device'] = self.ov_device
            try:
                model(blank, **kw)
            except Exception as exc:
                print(f"  [detector] warmup failed on {self.ov_device} "
                      f"({type(exc).__name__}); using CPU", flush=True)
                if is_openvino and self.ov_device != 'intel:cpu':
                    self.ov_device = 'intel:cpu'

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

    def _infer(self, model, frame, conf, iou_threshold, is_openvino):
        """Run one model; route OpenVINO to the iGPU with automatic CPU fallback."""
        if not is_openvino:
            return model(frame, conf=conf, iou=iou_threshold, verbose=False,
                         half=self.device != 'cpu', imgsz=self.imgsz)
        try:
            return model(frame, conf=conf, iou=iou_threshold, verbose=False,
                         imgsz=self.imgsz, device=self.ov_device)
        except Exception as exc:
            if self.ov_device != 'intel:cpu':
                print(f"  [detector] {self.ov_device} inference failed "
                      f"({type(exc).__name__}); falling back to intel:cpu", flush=True)
                self.ov_device = 'intel:cpu'
                return model(frame, conf=conf, iou=iou_threshold, verbose=False,
                             imgsz=self.imgsz, device=self.ov_device)
            raise

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None
    ) -> list[dict]:
        all_detections = []

        for model, config, is_openvino in self.models:
            conf = config.conf_threshold or conf_threshold
            results = self._infer(model, frame, conf, iou_threshold, is_openvino)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy, confs, clss = (_as_numpy(t) for t in (boxes.xyxy, boxes.conf, boxes.cls))
                for box, score, cls in zip(xyxy, confs, clss, strict=True):
                    conf_score = float(score)
                    orig_class_id = int(cls)
                    unified_class_id = orig_class_id + config.class_offset

                    if classes is not None and unified_class_id not in classes:
                        continue

                    if config.class_conf:
                        min_conf = config.class_conf.get(orig_class_id, conf)
                        if conf_score < min_conf:
                            continue

                    all_detections.append({
                        'box': [int(v) for v in box],
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
        classes: list[int] | None = None
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


def export_openvino(model_name: str = 'yolov8n.pt', imgsz: int = 640, half: bool = False):
    """Export a .pt model to OpenVINO IR. Returns the exported *_openvino_model dir path."""
    model = YOLO(model_name)
    out = model.export(format='openvino', imgsz=imgsz, half=half)
    print(f"Exported {model_name} → OpenVINO IR (imgsz={imgsz}) at {out}")
    return str(out)


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
    # person, bicycle, car, motorcycle, bus, truck, bird, cat, dog,
    # horse, sheep, cow, bear, backpack, umbrella, handbag, suitcase, license_plate
    'outdoor': [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 28, 80],
    'indoor': [0, 39, 41, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67, 73, 74, 81],
    'vehicles': [1, 2, 3, 5, 7, 80],
    'all': None,
}
