#!/bin/bash
# Installs and starts the VisionBox systemd service. Run once.
set -e
sudo install -m 644 /home/night/VisionBox/visionbox.service /etc/systemd/system/visionbox.service
sudo systemctl daemon-reload
sudo systemctl enable --now visionbox.service
sudo systemctl status --no-pager visionbox.service
