#!/usr/bin/env python3
"""Atomic model hot-swap and rollback for the live VisionBox detector.

promote(run_dir) copies the run's OpenVINO export into a durable slot under models/
(the active link never points into the volatile runs/ tree), repoints the
models/yolov8n_openvino_model symlink atomically and SIGHUPs visionbox.service, which
rebuilds the model under its lock and swaps it in without dropping a stream.
rollback() restores whatever was active before the last promote.

    promote_model.py runs/train/overnight_<ts>
    promote_model.py --rollback
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = os.path.join(ROOT, "models")
ACTIVE_LINK = os.path.join(MODELS_DIR, "yolov8n_openvino_model")
CANDIDATE_LINK = os.path.join(MODELS_DIR, "candidate")
PREV_ACTIVE = os.path.join(MODELS_DIR, ".prev_active")
SERVICE = "visionbox.service"


def find_export(run_dir=None):
    """Return the *_openvino_model directory under run_dir, else the staged candidate."""
    for search_root in filter(None, (run_dir, CANDIDATE_LINK)):
        base = os.path.realpath(search_root)
        if not os.path.exists(base):
            continue
        if os.path.isdir(base) and base.endswith("_openvino_model"):
            return base
        for dirpath, dirnames, _ in os.walk(base):
            for d in sorted(dirnames):
                if d.endswith("_openvino_model"):
                    return os.path.join(dirpath, d)
    raise FileNotFoundError(f"No *_openvino_model directory found under {run_dir or CANDIDATE_LINK}")


def _materialize(export_dir, run_dir=None):
    """Copy the export into a durable models/ slot; a no-op if it already lives there."""
    export_dir = os.path.abspath(export_dir)
    models_abs = os.path.abspath(MODELS_DIR)
    if export_dir == models_abs or export_dir.startswith(models_abs + os.sep):
        return export_dir
    if run_dir:
        tag = os.path.basename(os.path.normpath(run_dir))
    else:  # the staged candidate: name the slot after its training run, not after "best_openvino_model"
        parent = Path(export_dir).parent
        tag = parent.parent.name if parent.name == "weights" else parent.name
    slot = os.path.join(MODELS_DIR, f"{tag}_openvino_model")
    # Never overwrite the currently-active model dir while copying.
    if os.path.realpath(slot) == os.path.realpath(ACTIVE_LINK):
        slot += ".new"
    if os.path.exists(slot):
        shutil.rmtree(slot)
    shutil.copytree(export_dir, slot)
    return slot


def _atomic_symlink(target, link_path):
    tmp = f"{link_path}.tmp.{os.getpid()}"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.abspath(target), tmp)
    os.replace(tmp, link_path)


def _record_prev_active():
    """Persist the current active target to .prev_active for rollback."""
    if os.path.islink(ACTIVE_LINK):
        current = os.path.realpath(ACTIVE_LINK)
    elif os.path.exists(ACTIVE_LINK):
        current = os.path.abspath(ACTIVE_LINK)
    else:
        current = ""
    with open(PREV_ACTIVE, "w") as fh:
        fh.write(current + "\n")
    return current


def _reload():
    """SIGHUP the service for an in-place hot-swap; a restart would drop every camera stream."""
    try:
        out = subprocess.run(["systemctl", "show", SERVICE, "-p", "MainPID", "--value"],
                             capture_output=True, text=True, timeout=5, check=False)
        pid = out.stdout.strip()
        if pid.isdigit() and int(pid) > 0:
            os.kill(int(pid), signal.SIGHUP)
            return "sighup"
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        print(f"SIGHUP reload failed ({exc})")
    print(f"WARN: could not signal {SERVICE}; reload manually with: sudo systemctl reload-or-restart {SERVICE}")
    return "manual"


def promote(run_dir=None):
    """Promote the exported model under run_dir to the active slot (durable copy)."""
    export_dir = find_export(run_dir)
    durable = _materialize(export_dir, run_dir)
    prev = _record_prev_active()
    if prev and os.path.realpath(durable) == os.path.realpath(prev):
        print(f"model {durable} already active; nothing to promote")
        return durable
    _atomic_symlink(durable, ACTIVE_LINK)
    method = _reload()
    print(f"promoted {durable} -> {ACTIVE_LINK} (reload via {method})")
    return durable


def rollback():
    """Restore the model target recorded before the last promote."""
    if not os.path.exists(PREV_ACTIVE):
        raise FileNotFoundError(f"no {PREV_ACTIVE} to roll back to")
    with open(PREV_ACTIVE) as fh:
        prev = fh.read().strip()
    if not prev:
        raise ValueError(f"{PREV_ACTIVE} is empty; nothing to roll back to")
    if not os.path.exists(prev):
        raise FileNotFoundError(f"previous model {prev} no longer exists")
    _atomic_symlink(prev, ACTIVE_LINK)
    method = _reload()
    print(f"rolled back to {prev} (reload via {method})")
    return prev


def main(argv=None):
    parser = argparse.ArgumentParser(description="Promote or roll back the active VisionBox model.")
    parser.add_argument("run_dir", nargs="?",
                        help="Run directory containing the exported *_openvino_model (default: the staged candidate)")
    parser.add_argument("--rollback", action="store_true", help="Restore the previously active model")
    args = parser.parse_args(argv)

    if args.rollback:
        rollback()
    else:
        promote(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
