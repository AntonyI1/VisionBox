#!/usr/bin/env bash
# Install the VisionBox overnight self-training timer.
#   Run from the repo root:  sudo bash ./install-training-timer.sh
# visionbox-train.service is a template: this fills in the invoking user, group and checkout path.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_USER=${SUDO_USER:-$(id -un)}
RUN_GROUP=$(id -gn "$RUN_USER")

chmod +x "$REPO/scripts/training/run_overnight.sh"
sed -e "s#/opt/visionbox#$REPO#g" -e "s/^User=.*/User=$RUN_USER/" -e "s/^Group=.*/Group=$RUN_GROUP/" \
  "$REPO/visionbox-train.service" > /etc/systemd/system/visionbox-train.service
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
