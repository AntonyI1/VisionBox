#!/bin/bash
# Installs and starts the VisionBox systemd service. Run once from the repo root.
# visionbox.service hardcodes User=, Group=, WorkingDirectory= and the venv path:
# edit it first if this checkout lives elsewhere or runs as another user.
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIT_DIR=$(sed -n 's/^WorkingDirectory=//p' "$REPO/visionbox.service")
if [ "$UNIT_DIR" != "$REPO" ]; then
  echo "visionbox.service has WorkingDirectory=$UNIT_DIR but this checkout is $REPO; edit the unit first." >&2
  exit 1
fi
sudo install -m 644 "$REPO/visionbox.service" /etc/systemd/system/visionbox.service
sudo systemctl daemon-reload
sudo systemctl enable --now visionbox.service
sudo systemctl status --no-pager visionbox.service
