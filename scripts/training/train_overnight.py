"""Overnight self-training orchestrator and entrypoint for VisionBox.

Assembles the trainable dataset, fine-tunes yolov8n on CPU with the pretrained
head frozen, runs a base-class regression gate against COCO replay, and stages
(or promotes) the resulting OpenVINO model.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import assemble_dataset  # noqa: E402

BASE_WEIGHTS = "/home/night/VisionBox/yolov8n.pt"
SRC_DIR = "/home/night/VisionBox/src"
RUNS_ROOT = "/home/night/VisionBox/runs/train"
MODELS_DIR = "/home/night/VisionBox/models"
ACTIVE_MODEL = os.path.join(MODELS_DIR, "yolov8n_openvino_model")
CANDIDATE_LINK = os.path.join(MODELS_DIR, "candidate")
BASE_YAML = "/mnt/storage/visionbox/datasets/coco_replay/base.yaml"

MIN_TRAIN_IMGS = 50
MAX_EPOCHS = 50
IMGSZ = 640
REGRESSION_TOLERANCE = 0.01


def _map5095(weights, data_yaml):
    """Return base-class mAP50-95 for a model evaluated on the replay val set."""
    from ultralytics import YOLO

    metrics = YOLO(weights).val(
        data=data_yaml, device="cpu", imgsz=IMGSZ, verbose=False
    )
    return float(metrics.box.map)


def regression_gate(old_model, new_pt, base_yaml):
    """Compare base-class mAP50-95 of the active model vs the freshly trained one.

    Returns (passed, mAP_old, mAP_new). A missing replay set fails CLOSED
    (passed=False) so auto-promote never ships an unvalidated model. A missing
    active model (first-ever run) passes; otherwise passes iff new >= old - tol.
    """
    if not os.path.exists(base_yaml):
        # No replay baseline -> cannot prove base classes didn't regress.
        # Fail closed; the run still stages a candidate for manual review.
        return False, None, None

    map_new = _map5095(new_pt, base_yaml)

    if not os.path.exists(old_model):
        return True, None, map_new

    map_old = _map5095(old_model, base_yaml)
    passed = map_new >= map_old - REGRESSION_TOLERANCE
    return passed, map_old, map_new


def _stage_candidate(openvino_dir):
    if os.path.islink(CANDIDATE_LINK) or os.path.exists(CANDIDATE_LINK):
        os.unlink(CANDIDATE_LINK)
    os.symlink(openvino_dir, CANDIDATE_LINK)


def _write_report(run_dir, report):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "REPORT.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def train(run_ts, epochs, data_yaml):
    from ultralytics import YOLO

    model = YOLO(BASE_WEIGHTS)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        freeze=10,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        imgsz=IMGSZ,
        batch=16,
        device="cpu",
        patience=15,
        cache="disk",
        project=RUNS_ROOT,
        name=f"overnight_{run_ts}",
    )
    return str(results.save_dir)


def run(run_ts, epochs, auto_promote):
    os.makedirs(RUNS_ROOT, exist_ok=True)

    dataset = assemble_dataset.emit()
    n_train = dataset["n_train"]
    n_val = dataset["n_val"]

    run_dir = os.path.join(RUNS_ROOT, f"overnight_{run_ts}")

    if n_train < MIN_TRAIN_IMGS:
        report = {
            "gate": "ABORT",
            "reason": f"n_train {n_train} < MIN_TRAIN_IMGS {MIN_TRAIN_IMGS}",
            "mAP_old": None,
            "mAP_new": None,
            "n_train": n_train,
            "n_val": n_val,
            "promoted": False,
        }
        _write_report(run_dir, report)
        print(f"[train_overnight] ABORT: {report['reason']}", flush=True)
        return report

    save_dir = train(run_ts, epochs, dataset["data_yaml"])
    best_pt = os.path.join(save_dir, "weights", "best.pt")

    passed, map_old, map_new = regression_gate(ACTIVE_MODEL, best_pt, BASE_YAML)
    gate = "PASS" if passed else "FAIL"

    promoted = False
    if passed:
        sys.path.insert(0, SRC_DIR)
        from visionbox.detector_v2 import export_openvino

        openvino_dir = export_openvino(best_pt, imgsz=IMGSZ)
        _stage_candidate(openvino_dir)
        print(f"[train_overnight] staged candidate -> {openvino_dir}", flush=True)

        if auto_promote:
            import promote_model

            promote_model.promote(save_dir)
            promoted = True
            print("[train_overnight] auto-promoted new model", flush=True)

    report = {
        "gate": gate,
        "mAP_old": map_old,
        "mAP_new": map_new,
        "n_train": n_train,
        "n_val": n_val,
        "promoted": promoted,
    }
    _write_report(save_dir, report)
    print(f"[train_overnight] gate={gate} mAP_old={map_old} mAP_new={map_new}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overnight CPU fine-tune of yolov8n with regression gate and staging."
    )
    parser.add_argument(
        "--run-ts",
        required=True,
        help="Run timestamp/identifier supplied by the caller (used in the run name).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Training epochs (clamped to <= {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="On a passing gate, promote the new model live instead of only staging it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    epochs = max(1, min(args.epochs, MAX_EPOCHS))
    run(args.run_ts, epochs, args.auto_promote)


if __name__ == "__main__":
    main()
