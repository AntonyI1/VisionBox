#!/usr/bin/env bash
# VisionBox network hardening, part 2 — make Frigate's Docker ports tailnet-only.
#   Review, then run:  sudo bash /home/night/VisionBox/harden-frigate.sh
#
# PROBLEM: Docker publishes Frigate's 5000 (UI, no auth) / 8554 (RTSP) / 8555 (WebRTC)
# on 0.0.0.0 and DNATs traffic in the FORWARD path, which BYPASSES ufw's INPUT rules —
# so harden-network.sh never protected them and anyone on the LAN can open Frigate.
#
# FIX: filter in the DOCKER-USER chain (the supported hook Docker evaluates first for
# all forwarded container traffic), installed via /etc/ufw/after.rules so it persists
# across reboots and `ufw reload`. Compared to rebinding ports in docker-compose.yml,
# this needs no container restart and has no tailscale0-not-up-yet boot-order failure.
#
# Allowed:  tailnet (tailscale0), inter-container, host/localhost (VisionBox's
#           rtsp://127.0.0.1:8554 pulls go host->bridge and never traverse FORWARD).
# Blocked:  everything else (LAN, etc.) to 5000/8554/8555.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo bash $0"; exit 1; }

RULES=/etc/ufw/after.rules
MARKER='VISIONBOX-FRIGATE-DOCKER-USER'

if grep -q "$MARKER" "$RULES"; then
  echo "Rules already installed in $RULES — nothing to do."
else
  cp -a "$RULES" "${RULES}.bak-$(date +%Y%m%d_%H%M%S)"
  cat >> "$RULES" <<'EOF'

# --- VISIONBOX-FRIGATE-DOCKER-USER (added by harden-frigate.sh) ---
# NOTE: DNAT runs before FORWARD, so --dports below match the CONTAINER ports.
# They equal the host ports in docker-compose.yml; keep them in sync if that changes.
*filter
:DOCKER-USER - [0:0]
-A DOCKER-USER -m conntrack --ctstate RELATED,ESTABLISHED -j RETURN
-A DOCKER-USER -i tailscale0 -j RETURN
-A DOCKER-USER -i br-+ -j RETURN
-A DOCKER-USER -i docker0 -j RETURN
-A DOCKER-USER -p tcp -m multiport --dports 5000,8554,8555 -j DROP
-A DOCKER-USER -p udp --dport 8555 -j DROP
-A DOCKER-USER -j RETURN
COMMIT
# --- end VISIONBOX-FRIGATE-DOCKER-USER ---
EOF
  echo "Appended DOCKER-USER rules to $RULES (backup saved alongside)."
fi

ufw reload
echo
iptables -L DOCKER-USER -n --line-numbers
cat <<'NOTES'

--- DONE: Frigate (5000/8554/8555) is now tailnet-only. ---

VERIFY from a LAN device that is NOT on Tailscale — Frigate must NOT load:
    curl --max-time 4 http://192.168.1.252:5000/        # expect: timeout
Over Tailscale it must still work:
    curl --max-time 4 http://100.78.228.85:5000/        # expect: HTML
And VisionBox itself must still see its feeds (uses 127.0.0.1, unaffected):
    curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8085/   # expect: 401

ROLLBACK: restore the .bak file over /etc/ufw/after.rules and `ufw reload`.
NOTES
