#!/usr/bin/env python3
"""Download required models for VisionBox."""

from pathlib import Path

def main():
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    print("Downloading models...")

    # YOLOv8 (auto-downloads on first use, but we can trigger it)
    print("\n[1/2] YOLOv8n (COCO - 80 classes)...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✓ YOLOv8n ready")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # License plate model from HuggingFace
    print("\n[2/2] License plate detection model...")
    lp_path = models_dir / 'license-plate-finetune-v1n.pt'
    if lp_path.exists():
        print("  ✓ Already downloaded")
    else:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id='morsetechlab/yolov11-license-plate-detection',
                filename='license-plate-finetune-v1n.pt',
                local_dir=str(models_dir)
            )
            print("  ✓ License plate model downloaded")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print("    Install huggingface_hub: pip install huggingface_hub")

    print("\n" + "="*50)
    print("Setup complete! Run the demo:")
    print("  python scripts/camera_demo.py <camera_url>")
    print("="*50)


if __name__ == '__main__':
    main()
