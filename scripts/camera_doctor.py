#!/usr/bin/env python3
"""Camera doctor — run from cron every 5 minutes.

Probes each camera's own RTSP endpoint and its go2rtc restream. A camera that
fails 3 consecutive runs (~15 min) is declared DOWN: logged, optionally pushed
via ntfy (set NTFY_TOPIC in .env), and sent an ONVIF SystemReboot at most once
per hour. Soft hangs (RTSP daemon dead, ONVIF alive) self-heal this way; a full
firmware hang like front_garage's Sep 2026 one still needs a power/cloud reboot
but gets noticed in minutes instead of days. Probes are liveness-only: any RTSP
status line (401 included) counts as alive.
"""

import argparse
import base64
import fcntl
import hashlib
import json
import os
import signal
import socket
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

CAM_USER = os.environ.get('CAM_USER', '')
CAM_PASS = os.environ.get('CAM_PASS', '')
EMP_PASS = os.environ.get('EMPIRETECH_PASS', '')

# rtsp: the camera's own endpoint (isolates camera health from go2rtc/Frigate);
# restream: what Frigate + VisionBox actually consume; onvif: reboot target.
CAMERAS = {
    'front_door': {
        'rtsp': f'rtsp://admin:{EMP_PASS}@192.168.1.50:554/cam/realmonitor?channel=1&subtype=1',
        'restream': 'front_door_sub',
        'onvif': ('192.168.1.50', 80, 'admin', EMP_PASS),
    },
    'front_garage': {
        'rtsp': f'rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.249:554/stream2',
        'restream': 'front_garage',
        'onvif': ('192.168.1.249', 2020, CAM_USER, CAM_PASS),
    },
    'backyard': {
        'rtsp': f'rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.101:554/stream1',
        'restream': 'backyard',
        'onvif': ('192.168.1.101', 2020, CAM_USER, CAM_PASS),
    },
    'backyard_gate': {
        'rtsp': f'rtsp://{CAM_USER}:{CAM_PASS}@192.168.1.192:554/stream1',
        'restream': 'backyard_gate',
        'onvif': ('192.168.1.192', 2020, CAM_USER, CAM_PASS),
    },
}

STATE_FILE = ROOT / 'logs' / 'camera_doctor_state.json'
FAILS_BEFORE_DOWN = 3
REBOOT_COOLDOWN_S = 3600


def log(msg: str):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def rtsp_alive(url: str, timeout: float = 4.0) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 554),
                                      timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\n"
                          "Accept: application/sdp\r\nUser-Agent: camera-doctor\r\n\r\n").encode())
            return sock.recv(64).startswith(b'RTSP/1.0 ')
    except OSError:
        return False


def onvif_reboot(host: str, port: int, user: str, password: str) -> str:
    """WS-UsernameToken digest SystemReboot; returns a short result string."""
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Header>
    <Security s:mustUnderstand="1"
        xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
      <UsernameToken>
        <Username>{user}</Username>
        <Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>
        <Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>
        <Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>
      </UsernameToken>
    </Security>
  </s:Header>
  <s:Body><tds:SystemReboot/></s:Body>
</s:Envelope>"""
    req = urllib.request.Request(
        f'http://{host}:{port}/onvif/device_service', data=envelope.encode(),
        headers={'Content-Type': 'application/soap+xml; charset=utf-8'})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read(500).decode(errors='replace')
            return 'accepted' if ('RebootMessage' in body or resp.status == 200) \
                else f'http {resp.status}'
    except Exception as exc:
        return f'{type(exc).__name__}'


def notify(title: str, message: str):
    topic = os.environ.get('NTFY_TOPIC')
    if not topic:
        return
    base = os.environ.get('NTFY_URL', 'https://ntfy.sh').rstrip('/')
    try:
        req = urllib.request.Request(f'{base}/{topic}',
                                     data=message.encode(),
                                     headers={'Title': title})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:
        log(f"notify failed: {type(exc).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='print every probe result')
    args = parser.parse_args()

    signal.alarm(240)  # hard ceiling: a wedged probe pile-up must not outlive the cron interval

    STATE_FILE.parent.mkdir(exist_ok=True)
    lock = open(STATE_FILE.with_suffix('.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # previous run still going

    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        state = {}

    now = time.time()
    for name, cam in CAMERAS.items():
        st = state.setdefault(name, {'fails': 0, 'down_since': None,
                                     'last_reboot': 0, 'notified': False,
                                     'restream_fails': 0})
        alive = rtsp_alive(cam['rtsp'])
        if args.verbose:
            log(f"[{name}] camera={'up' if alive else 'DOWN'}")

        if alive:
            if st['down_since']:
                mins = (now - st['down_since']) / 60
                log(f"[{name}] RECOVERED after {mins:.0f} min")
                if st['notified']:
                    notify(f'VisionBox: {name} recovered',
                           f'{name} is answering again after {mins:.0f} min down.')
            st.update(fails=0, down_since=None, notified=False)
        else:
            st['fails'] += 1
            if st['fails'] == FAILS_BEFORE_DOWN:
                st['down_since'] = now - FAILS_BEFORE_DOWN * 300
                log(f"[{name}] DOWN ({FAILS_BEFORE_DOWN} consecutive probe failures)")
                notify(f'VisionBox: {name} DOWN',
                       f'{name} RTSP has not answered for ~15 min. '
                       f'Attempting ONVIF reboot; if it stays down it needs a power cycle.')
                st['notified'] = True
            if st['fails'] >= FAILS_BEFORE_DOWN and now - st['last_reboot'] > REBOOT_COOLDOWN_S:
                st['last_reboot'] = now
                result = onvif_reboot(*cam['onvif'])
                log(f"[{name}] ONVIF reboot attempt: {result}")

        # Restream health only means anything while the camera itself is up —
        # otherwise it's the same outage seen through go2rtc.
        if alive:
            r_ok = rtsp_alive(f"rtsp://127.0.0.1:8554/{cam['restream']}")
            if args.verbose:
                log(f"[{name}] restream={'up' if r_ok else 'DOWN'}")
            st['restream_fails'] = 0 if r_ok else st['restream_fails'] + 1
            if st['restream_fails'] == FAILS_BEFORE_DOWN:
                log(f"[{name}] camera up but go2rtc restream dead — check the frigate container")
                notify('VisionBox: restream problem',
                       f'{name} camera is fine but its go2rtc restream is not answering. '
                       f'Frigate/go2rtc likely needs a restart.')

    STATE_FILE.write_text(json.dumps(state))

    # Keep the cron-appended log bounded.
    logfile = ROOT / 'logs' / 'camera_doctor.log'
    try:
        if logfile.stat().st_size > 1_000_000:
            lines = logfile.read_text().splitlines()[-500:]
            logfile.write_text('\n'.join(lines) + '\n')
    except OSError:
        pass


if __name__ == '__main__':
    main()
