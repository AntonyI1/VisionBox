"""Image preprocessing for YOLO inference."""

import cv2
import numpy as np
import torch


def letterbox(
    image: np.ndarray,
    target_size: int = 640
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image preserving aspect ratio with padding.

    Args:
        image: Input image (H, W, C) in BGR format
        target_size: Target square size

    Returns:
        resized: Letterboxed image (target_size, target_size, 3)
        scale: Scale factor applied
        padding: (pad_w, pad_h) padding added
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return canvas, scale, (pad_w, pad_h)


def preprocess(
    image: np.ndarray,
    device: str = 'cuda'
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """
    Preprocess image for YOLO inference.

    Args:
        image: Input image (H, W, C) in BGR format
        device: Target device ('cuda' or 'cpu')

    Returns:
        tensor: Preprocessed tensor (1, 3, 640, 640)
        scale: Scale factor for coordinate conversion
        padding: Padding for coordinate conversion
    """
    letterboxed, scale, padding = letterbox(image, 640)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = normalized.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device).contiguous()

    return tensor, scale, padding
