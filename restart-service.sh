#!/bin/bash
# Restart the VisionBox service and tail the log briefly.
set -e
sudo systemctl restart visionbox.service
sleep 8
sudo systemctl status --no-pager visionbox.service | head -10
echo "---"
journalctl -u visionbox.service --since "30 seconds ago" --no-pager \
  | grep -vE 'h264 @|error while decoding|left block unavailable|top block unavailable|requested intra' \
  | tail -20
