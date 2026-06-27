#!/usr/bin/env bash
# VisionBox network hardening — make host services tailnet-only + cut attack surface.
#   Review, then run:  sudo bash /home/night/VisionBox/harden-network.sh
#
# WHAT THIS DOES (host-level, reversible with `sudo ufw disable`):
#   * ufw: default-deny inbound / allow outbound; allow loopback; allow the whole
#     Tailscale interface (trusted tailnet); allow SSH (22) so you are never locked out.
#     => the VisionBox dashboard (host process on :8085) becomes reachable ONLY over
#        Tailscale (+ localhost), on top of the HTTP Basic auth already enabled.
#   * Disables rpcbind (port 111): the NFS mount is v4.2 (single TCP port, client-
#     initiated), so rpcbind is not needed — closing it removes a needless surface.
#
# NOT done here (they touch Frigate/Docker or need a choice) — see the notes at the end.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo bash $0"; exit 1; }

TS_IF=tailscale0

command -v ufw >/dev/null || { echo "installing ufw..."; apt-get update -qq && apt-get install -y ufw; }

# Allow SSH + the whole tailnet BEFORE enabling, so an active session can't be cut.
ufw allow in on lo
ufw allow 22/tcp comment 'SSH'
ufw allow in on "$TS_IF" comment 'Tailscale tailnet (trusted)'
ufw default deny incoming
ufw default allow outgoing
ufw --force enable
ufw status verbose

# rpcbind is unnecessary for NFSv4 — stop + disable to close port 111.
if systemctl is-active --quiet rpcbind || systemctl is-enabled --quiet rpcbind 2>/dev/null; then
  systemctl disable --now rpcbind.socket 2>/dev/null || true
  systemctl disable --now rpcbind 2>/dev/null || true
  echo "rpcbind disabled (NFS is v4.2; does not need it)."
fi

cat <<'NOTES'

--- DONE: host services are now tailnet-only (+ dashboard auth). ---

VERIFY from a LAN device that is NOT on Tailscale — the dashboard must NOT load:
    curl --max-time 4 http://192.168.1.252:8085/        # expect: hang / connection refused
On Tailscale it should prompt for the dashboard password.

STILL TO DO (manual — not done automatically):
  1. Frigate's :5000/:8554/:8555 are published by DOCKER, which BYPASSES ufw.
     To make Frigate tailnet-only, bind its compose ports to the Tailscale IP:
         ports:
           - "100.78.228.85:5000:5000"
           - "100.78.228.85:8554:8554"
           - "100.78.228.85:8555:8555/tcp"
           - "100.78.228.85:8555:8555/udp"
     then re-create:  docker compose up -d   (keeps frigate.antonyibrahim.com working over Tailscale)
  2. Optional SSH brute-force protection:
         sudo apt-get install -y fail2ban && sudo systemctl enable --now fail2ban
  3. Confirm your home router has NO port-forward to 192.168.1.252 (esp. 8085 / 5000 / 22).
NOTES
