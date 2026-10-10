#!/bin/bash
# Installs and starts the VisionBox systemd service. Run once from the repo root.
# visionbox.service is a template: this fills in the invoking user, group and checkout path.
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_USER=${SUDO_USER:-$(id -un)}
RUN_GROUP=$(id -gn "$RUN_USER")
UNIT=$(mktemp)
trap 'rm -f "$UNIT"' EXIT
sed -e "s#/opt/visionbox#$REPO#g" -e "s/^User=.*/User=$RUN_USER/" -e "s/^Group=.*/Group=$RUN_GROUP/" \
  "$REPO/visionbox.service" > "$UNIT"
sudo install -m 644 "$UNIT" /etc/systemd/system/visionbox.service
sudo systemctl daemon-reload
sudo systemctl enable --now visionbox.service
sudo systemctl status --no-pager visionbox.service
