#!/usr/bin/env python3
"""Atomic model hot-swap and rollback for VisionBox.

promote(run_dir) copies the freshly exported OpenVINO model into a DURABLE slot
under models/, repoints the active-model symlink at it, and asks the live
detector to hot-swap via SIGHUP. rollback() restores whatever target was active
before the last promote.

Design choices that keep live surveillance safe:
  * The active link (models/yolov8n_openvino_model) is NEVER pointed into the
    volatile runs/train tree — we copy the export into models/ first, so a later
    cleanup of runs/ can't dangle the live model.
  * Reload is via SIGHUP (an on-box signal needing no auth); the service rebuilds
    the model under its lock, pausing the cameras for a single frame, not a whole
    restart. We never auto-restart the service (that would drop every stream).
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys

MODELS_DIR = "/home/night/VisionBox/models"
ACTIVE_LINK = os.path.join(MODELS_DIR, "yolov8n_openvino_model")
CANDIDATE_LINK = os.path.join(MODELS_DIR, "candidate")
PREV_ACTIVE = os.path.join(MODELS_DIR, ".prev_active")
SERVICE = "visionbox.service"


def find_export(run_dir=None):
    """Return the *_openvino_model directory to promote.

    Looks under run_dir first, then falls back to the staged candidate link.
    """
    search_roots = []
    if run_dir:
        search_roots.append(run_dir)
    search_roots.append(CANDIDATE_LINK)

    for root in search_roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if not os.path.exists(root):
            continue
        if os.path.isdir(root) and root.endswith("_openvino_model"):
            return root
        for dirpath, dirnames, _ in os.walk(root):
            for d in sorted(dirnames):
                if d.endswith("_openvino_model"):
                    return os.path.join(dirpath, d)
    raise FileNotFoundError(
        "No *_openvino_model directory found under %s"
        % (run_dir or CANDIDATE_LINK)
    )


def _materialize(export_dir, run_dir=None):
    """Copy the export into a durable models/ slot so the active link never depends
    on the volatile runs/train tree. Idempotent if the export already lives under
    models/. Returns the durable path."""
    export_dir = os.path.abspath(export_dir)
    models_abs = os.path.abspath(MODELS_DIR)
    if export_dir == models_abs or export_dir.startswith(models_abs + os.sep):
        return export_dir  # already durable (e.g. the base model itself)
    tag = os.path.basename(os.path.normpath(run_dir)) if run_dir else os.path.basename(export_dir)
    slot = os.path.join(MODELS_DIR, "%s_openvino_model" % tag)
    # Never overwrite the currently-active model dir while copying.
    if os.path.realpath(slot) == os.path.realpath(ACTIVE_LINK):
        slot += ".new"
    if os.path.exists(slot):
        shutil.rmtree(slot)
    shutil.copytree(export_dir, slot)
    return slot


def _atomic_symlink(target, link_path):
    """Point link_path at target atomically via temp symlink + os.replace."""
    target = os.path.abspath(target)
    tmp = link_path + ".tmp.%d" % os.getpid()
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target, tmp)
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
    """Hot-swap the new model into the live detector by SIGHUP-ing the service.

    The service's handler rebuilds the model under its lock (cameras pause for a
    single frame, not the whole rebuild). We deliberately do NOT auto-restart the
    service — that would drop every camera stream — so on failure we just print how
    to reload manually.
    """
    try:
        out = subprocess.run(
            ["systemctl", "show", SERVICE, "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=5,
        )
        pid = out.stdout.strip()
        if pid.isdigit() and int(pid) > 0:
            os.kill(int(pid), signal.SIGHUP)
            return "sighup"
    except (subprocess.SubprocessError, ProcessLookupError, PermissionError,
            OSError, ValueError) as exc:
        print("SIGHUP reload failed (%s)" % exc)
    print("WARN: could not signal %s; reload manually with: "
          "sudo systemctl reload-or-restart %s" % (SERVICE, SERVICE))
    return "manual"


def promote(run_dir=None):
    """Promote the exported model under run_dir to the active slot (durable copy)."""
    export_dir = find_export(run_dir)
    durable = _materialize(export_dir, run_dir)
    prev = _record_prev_active()
    if prev and os.path.realpath(durable) == os.path.realpath(prev):
        print("model %s already active; nothing to promote" % durable)
        return durable
    _atomic_symlink(durable, ACTIVE_LINK)
    method = _reload()
    print("promoted %s -> %s (reload via %s)" % (durable, ACTIVE_LINK, method))
    return durable


def rollback():
    """Restore the model target recorded before the last promote."""
    if not os.path.exists(PREV_ACTIVE):
        raise FileNotFoundError("no %s to roll back to" % PREV_ACTIVE)
    with open(PREV_ACTIVE) as fh:
        prev = fh.read().strip()
    if not prev:
        raise ValueError("%s is empty; nothing to roll back to" % PREV_ACTIVE)
    if not os.path.exists(prev):
        raise FileNotFoundError("previous model %s no longer exists" % prev)
    _atomic_symlink(prev, ACTIVE_LINK)
    method = _reload()
    print("rolled back to %s (reload via %s)" % (prev, method))
    return prev


def main(argv=None):
    parser = argparse.ArgumentParser(description="Promote or roll back the active VisionBox model.")
    parser.add_argument("run_dir", nargs="?", help="Run directory containing the exported *_openvino_model")
    parser.add_argument("--rollback", action="store_true", help="Restore the previously active model")
    args = parser.parse_args(argv)

    if args.rollback:
        rollback()
        return 0
    if not args.run_dir:
        parser.error("run_dir is required unless --rollback is given")
    promote(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
