#!/usr/bin/env bash
# Install the VisionBox overnight self-training timer.
#   Run from the repo root:  sudo bash ./install-training-timer.sh
# visionbox-train.service hardcodes User=, Group=, WorkingDirectory= and the script
# path: edit it first if this checkout lives elsewhere or runs as another user.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIT_DIR=$(sed -n 's/^WorkingDirectory=//p' "$REPO/visionbox-train.service")
if [ "$UNIT_DIR" != "$REPO" ]; then
  echo "visionbox-train.service has WorkingDirectory=$UNIT_DIR but this checkout is $REPO; edit the units first." >&2
  exit 1
fi

chmod +x "$REPO/scripts/training/run_overnight.sh"
cp "$REPO/visionbox-train.service" /etc/systemd/system/visionbox-train.service
cp "$REPO/visionbox-train.timer"   /etc/systemd/system/visionbox-train.timer
systemctl daemon-reload
systemctl enable --now visionbox-train.timer

echo "--- timer installed ---"
systemctl list-timers visionbox-train.timer --no-pager || true
echo
echo "Run one now (stage-only, watch live):"
echo "  sudo systemctl start visionbox-train.service && journalctl -u visionbox-train.service -f"
echo "Enable auto-promote: uncomment Environment=AUTO_PROMOTE=1 in"
echo "  /etc/systemd/system/visionbox-train.service  then  sudo systemctl daemon-reload"
