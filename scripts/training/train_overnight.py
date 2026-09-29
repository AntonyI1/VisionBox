#!/usr/bin/env python3
"""Overnight self-training: assemble the dataset, fine-tune yolov8n on CPU with the backbone
frozen, gate the result on base-class mAP against the COCO replay set, then stage the
OpenVINO export as models/candidate (or promote it live with --auto-promote).

Invoked by run_overnight.sh from visionbox-train.timer; writes REPORT.json into the run dir.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import assemble_dataset
import promote_model

RUNS_ROOT = ROOT / "runs" / "train"

MIN_TRAIN_IMGS = 50
MAX_EPOCHS = 50
IMGSZ = 640
REGRESSION_TOLERANCE = 0.01


def _map5095(weights, data_yaml):
    """Base-class mAP50-95 of a model on the replay set."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(data_yaml), device="cpu", imgsz=IMGSZ, verbose=False)
    return float(metrics.box.map)


def regression_gate(old_model, new_pt, base_yaml):
    """Compare base-class mAP50-95 of the active model against the candidate.

    Returns (passed, mAP_old, mAP_new). A missing replay set fails CLOSED so auto-promote
    never ships an unvalidated model; a missing active model (first run) passes.
    """
    if not os.path.exists(base_yaml):
        return False, None, None
    map_new = _map5095(new_pt, base_yaml)
    if not os.path.exists(old_model):
        return True, None, map_new
    map_old = _map5095(old_model, base_yaml)
    return map_new >= map_old - REGRESSION_TOLERANCE, map_old, map_new


def _stage_candidate(openvino_dir):
    link = promote_model.CANDIDATE_LINK
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(openvino_dir, link)


def _write_report(run_dir, report):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "REPORT.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def train(run_ts, epochs, data_yaml):
    from ultralytics import YOLO

    results = YOLO(str(assemble_dataset.BASE_WEIGHTS)).train(
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
        project=str(RUNS_ROOT),
        name=f"overnight_{run_ts}",
    )
    return str(results.save_dir)


def run(run_ts, epochs, auto_promote):
    os.makedirs(RUNS_ROOT, exist_ok=True)

    dataset = assemble_dataset.emit()
    n_train, n_val = dataset["n_train"], dataset["n_val"]
    run_dir = RUNS_ROOT / f"overnight_{run_ts}"

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

    passed, map_old, map_new = regression_gate(
        promote_model.ACTIVE_LINK, best_pt, assemble_dataset.COCO_REPLAY_YAML
    )
    gate = "PASS" if passed else "FAIL"

    promoted = False
    if passed:
        from visionbox.detector_v2 import export_openvino

        openvino_dir = export_openvino(best_pt, imgsz=IMGSZ)
        _stage_candidate(openvino_dir)
        print(f"[train_overnight] staged candidate -> {openvino_dir}", flush=True)
        if auto_promote:
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
    run(args.run_ts, max(1, min(args.epochs, MAX_EPOCHS)), args.auto_promote)


if __name__ == "__main__":
    main()
