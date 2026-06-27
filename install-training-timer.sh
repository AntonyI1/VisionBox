#!/usr/bin/env bash
# Install the VisionBox overnight self-training timer.
#   Run with:  sudo bash /home/night/VisionBox/install-training-timer.sh
set -euo pipefail
REPO=/home/night/VisionBox

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
