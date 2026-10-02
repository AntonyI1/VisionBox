#!/usr/bin/env python3
"""Camera doctor — run from cron every 5 minutes.

Part 1, cameras: probes each camera's own RTSP endpoint and its go2rtc restream.
A camera that fails 3 consecutive runs (~15 min) is declared DOWN: logged,
optionally pushed via ntfy (set NTFY_TOPIC in .env), and sent an ONVIF
SystemReboot at most once per hour. Soft hangs (RTSP daemon dead, ONVIF alive)
self-heal this way; a full firmware hang like front_garage's Sep 2026 one still
needs a power/cloud reboot but gets noticed in minutes instead of days. Probes
are liveness-only: any RTSP status line (401 included) counts as alive.

Part 2, Frigate pipeline watchdog (added 2026-09-09 after two same-day silent
failures — the iGPU boot race and an NFS stale-handle that killed Frigate's
detected_frames_processor thread — both left the container "healthy" while it
recorded nothing for hours). It looks only at Frigate-internal liveness signals,
never at whether motion or recordings happened, so a quiet camera can never
trigger it:
  * a dead thread ("Exception in thread …") or VAAPI device errors in the
    Frigate log                                             -> restart now
  * segments piling up in Frigate's cache ("Too many unprocessed recording
    segments"), latest.jpg frozen on a camera that reports frames, a camera
    whose restream is up but Frigate reads 0 fps, or an unreachable API
                                                            -> one strike;
    two consecutive strikes (~10 min) or two distinct symptoms at once
                                                            -> restart
Guards: a container younger than 10 min is left alone; no restart while the
NFS share is unmounted or stale (Frigate would just re-fail — alert instead);
30 min cooldown between auto-restarts; at most 3 per 6 h, then alert-only.
`--dry-run` prints decisions without restarting or notifying. Tunables can be
overridden from .env: FRIGATE_API, FRIGATE_CONTAINER, FRIGATE_NFS_MOUNT,
DOCTOR_WARMUP_S, DOCTOR_RESTART_COOLDOWN_S.
"""

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
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


DRY_RUN = False


def notify(title: str, message: str):
    topic = os.environ.get('NTFY_TOPIC')
    if not topic or DRY_RUN:
        return
    base = os.environ.get('NTFY_URL', 'https://ntfy.sh').rstrip('/')
    try:
        req = urllib.request.Request(f'{base}/{topic}',
                                     data=message.encode(),
                                     headers={'Title': title})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:
        log(f"notify failed: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Frigate pipeline watchdog
# ---------------------------------------------------------------------------
FRIGATE_API = os.environ.get('FRIGATE_API', 'http://127.0.0.1:5000').rstrip('/')
FRIGATE_CONTAINER = os.environ.get('FRIGATE_CONTAINER', 'frigate')
NFS_MOUNT = Path(os.environ.get('FRIGATE_NFS_MOUNT', '/mnt/storage'))
NFS_PROBE_DIRS = ('frigate/clips', 'frigate/recordings')
WARMUP_S = int(os.environ.get('DOCTOR_WARMUP_S', 600))
RESTART_COOLDOWN_S = int(os.environ.get('DOCTOR_RESTART_COOLDOWN_S', 1800))
RESTART_BUDGET_N, RESTART_BUDGET_WINDOW_S = 3, 6 * 3600
UNPROCESSED_SUSTAINED = 6   # maintainer warnings per run that mean "stuck", not a blip
LOG_WINDOW_S = 6 * 60       # first-run lookback; later runs continue from a cursor
ALERT_REPEAT_S = 3600
DOCKER = '/usr/bin/docker'

TS_RE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
LOG_SIGNATURES = {
    'dead_thread': re.compile(r'Exception in thread (\S+?):'),
    'unprocessed': re.compile(r'Too many unprocessed recording segments in cache for (\w+)'),
    'estale': re.compile(r'Stale file handle'),
    'va_error': re.compile(r'No VA display found|Hardware device setup failed'),
}


def http_get(url: str, timeout: float = 10) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def container_state() -> tuple[str, float]:
    """(docker status, epoch the container last started). status 'unknown' if docker fails."""
    try:
        out = subprocess.run([DOCKER, 'inspect', '-f', '{{.State.Status}} {{.State.StartedAt}}',
                              FRIGATE_CONTAINER], capture_output=True, text=True, timeout=20)
        status, _, started = out.stdout.strip().partition(' ')
        if out.returncode != 0 or not status:
            return 'unknown', 0.0
        head, _, frac = started.rstrip('Z').partition('.')
        base = datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        return status, base + (float('0.' + frac[:6]) if frac else 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 'unknown', 0.0


def parse_frigate_log(lines, since: float) -> dict:
    """Count failure signatures in Frigate log lines newer than `since` (epoch).

    Every line from /api/logs starts with a UTC timestamp; a line without one
    (shouldn't happen) inherits the previous timestamp.
    """
    found = {'dead_thread': [], 'unprocessed': {}, 'estale': 0, 'va_error': 0, 'newest': since}
    ts = None
    for line in lines:
        m = TS_RE.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') \
                    .replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        if ts is None or ts <= since:
            continue
        found['newest'] = max(found['newest'], ts)
        for key, rx in LOG_SIGNATURES.items():
            mm = rx.search(line)
            if not mm:
                continue
            if key == 'dead_thread':
                found['dead_thread'].append(mm.group(1))
            elif key == 'unprocessed':
                found['unprocessed'][mm.group(1)] = found['unprocessed'].get(mm.group(1), 0) + 1
            else:
                found[key] += 1
            break
    return found


def nfs_write_probe(directory: Path, timeout: float = 20) -> str:
    """'ok', 'hung', or the OSError text. Runs in a thread so a dead NAS can't wedge the run."""
    result = {}

    def _probe():
        probe = directory / f'.camera_doctor_probe_{os.getpid()}'
        try:
            probe.write_text('probe')
            probe.unlink()
            result['ok'] = True
        except OSError as exc:
            result['err'] = f'errno {exc.errno} {exc.strerror}'

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return 'hung'
    return 'ok' if result.get('ok') else result.get('err', 'unknown error')


def frame_hash(camera: str):
    try:
        return hashlib.md5(http_get(f'{FRIGATE_API}/api/{camera}/latest.jpg?h=180')).hexdigest()
    except Exception:
        return None


def decide(immediate: dict, symptoms: dict, strikes: int) -> tuple[bool, int]:
    """Pure policy: (restart?, new strike count)."""
    if symptoms:
        strikes += 1
    else:
        strikes = 0
    restart = bool(immediate) or (bool(symptoms) and (strikes >= 2 or len(symptoms) >= 2))
    return restart, strikes


def restart_blocked(fs: dict, now: float, nfs_problems: list) -> str:
    """'' if an auto-restart may proceed, else the reason it must not."""
    recent = [t for t in fs.get('restarts', []) if now - t < RESTART_BUDGET_WINDOW_S]
    fs['restarts'] = recent
    if nfs_problems:
        return 'NFS share unhealthy — a restart would just fail again'
    if recent and now - recent[-1] < RESTART_COOLDOWN_S:
        return f'cooldown — last auto-restart {(now - recent[-1]) / 60:.0f} min ago'
    if len(recent) >= RESTART_BUDGET_N:
        return f'budget — already {len(recent)} auto-restarts in 6 h'
    return ''


def frigate_watchdog(state: dict, restream_up: dict, now: float, verbose: bool):
    fs = state.setdefault('frigate', {})
    fs.setdefault('strikes', 0)
    fs.setdefault('restarts', [])
    fs.setdefault('frame_hash', {})
    fs.setdefault('alerts', {})
    fs.setdefault('down_runs', 0)

    def alert(key: str, title: str, message: str):
        if now - fs['alerts'].get(key, 0) < ALERT_REPEAT_S:
            return
        fs['alerts'][key] = now
        log(f"[frigate] ALERT {title}: {message}")
        notify(title, message)

    status, started = container_state()
    if status != 'running':
        fs['down_runs'] += 1
        fs['strikes'] = 0
        if fs['down_runs'] >= 2:
            alert('not_running', 'VisionBox: Frigate container not running',
                  f"docker reports the frigate container is '{status}'. Docker's restart "
                  f"policy should bring it back; check `docker ps -a` if it stays down.")
        return
    fs['down_runs'] = 0
    if now - started < WARMUP_S:
        if verbose:
            log(f"[frigate] container is {(now - started) / 60:.1f} min old — skipping pipeline checks")
        fs.update(strikes=0, frame_hash={})
        return

    # The NFS share: nothing Frigate does can work on a stale or missing export,
    # and restarting it would just reproduce the failure.
    nfs_problems = []
    if not os.path.ismount(NFS_MOUNT):
        nfs_problems.append(f'{NFS_MOUNT} is not mounted')
    else:
        for sub in NFS_PROBE_DIRS:
            res = nfs_write_probe(NFS_MOUNT / sub)
            if res != 'ok':
                nfs_problems.append(f'{sub}: {res}')
    if nfs_problems:
        alert('nfs', 'VisionBox: NFS share problem',
              '; '.join(nfs_problems) + '. Frigate cannot record until the NAS export is '
              'healthy; not restarting it.')

    immediate, symptoms = {}, {}

    try:
        cams = json.loads(http_get(f'{FRIGATE_API}/api/stats')).get('cameras', {})
    except Exception as exc:
        cams = None
        symptoms['api_down'] = f'/api/stats unreachable ({type(exc).__name__})'

    cursor = fs.get('log_cursor')
    since = now - LOG_WINDOW_S if cursor is None else max(cursor, now - 3600)
    try:
        payload = json.loads(http_get(f'{FRIGATE_API}/api/logs/frigate?start=-1500', timeout=15))
        found = parse_frigate_log(payload.get('lines', []), since)
        fs['log_cursor'] = found['newest']
    except Exception as exc:
        found = None
        symptoms.setdefault('api_down', f'/api/logs unreachable ({type(exc).__name__})')

    if found:
        if found['dead_thread']:
            immediate['dead_thread'] = 'dead thread: ' + ', '.join(sorted(set(found['dead_thread'])))
        if found['va_error']:
            immediate['va_error'] = (f"{found['va_error']} VAAPI device errors "
                                     f"(GPU render node missing from the container?)")
        unproc = sum(found['unprocessed'].values())
        if unproc >= UNPROCESSED_SUSTAINED:
            symptoms['unprocessed'] = (f"{unproc} 'unprocessed recording segments' warnings "
                                       f"({', '.join(sorted(found['unprocessed']))})")
        if found['estale']:
            alert('estale', 'VisionBox: NFS stale file handles',
                  f"Frigate logged {found['estale']} 'Stale file handle' errors — the NAS "
                  f"export changed underneath it. Watching for a dead thread.")

    if cams is not None:
        try:
            cfg_cams = json.loads(http_get(f'{FRIGATE_API}/api/config')).get('cameras', {})
        except Exception:
            cfg_cams = {}
        enabled = [c for c in CAMERAS if cfg_cams.get(c, {}).get('enabled', True)]
        fps = {c: (cams.get(c) or {}).get('camera_fps') or 0 for c in enabled}
        zero = [c for c in enabled if restream_up.get(c) and fps[c] == 0]
        if zero:
            symptoms['zero_fps'] = 'restream up but Frigate reads 0 fps: ' + ', '.join(zero)
        frozen, hashes = [], {}
        for c in enabled:
            if fps[c] <= 0:
                continue
            h = frame_hash(c)
            if h is None:
                continue
            hashes[c] = h
            if fs['frame_hash'].get(c) == h:
                frozen.append(c)
        fs['frame_hash'] = hashes
        if frozen:
            symptoms['frozen'] = ('latest.jpg unchanged for 5 min on a camera reporting frames: '
                                  + ', '.join(frozen))

    restart, fs['strikes'] = decide(immediate, symptoms, fs['strikes'])
    if verbose:
        log(f"[frigate] immediate={immediate or '-'} symptoms={symptoms or '-'} "
            f"strikes={fs['strikes']} restart={restart}")
    if not restart:
        if symptoms:
            log('[frigate] strike 1: ' + '; '.join(symptoms.values()))
        elif fs.pop('verify_after_restart', False):
            log('[frigate] healthy after auto-restart')
            notify('VisionBox: Frigate healthy again',
                   'Pipeline checks are clean after the auto-restart.')
        return

    why = '; '.join(list(immediate.values()) + list(symptoms.values()))
    blocked = restart_blocked(fs, now, nfs_problems)
    if blocked:
        log(f"[frigate] restart wanted ({why}) but not doing it: {blocked}")
        if blocked.startswith('budget'):
            alert('budget', 'VisionBox: Frigate keeps failing',
                  f'{why}. {blocked}; needs a human.')
        return

    log(f"[frigate] AUTO-RESTART: {why}")
    if DRY_RUN:
        log('[frigate] dry-run: not restarting')
        return
    # Persist the restart before issuing it so a killed run can't cause a loop.
    fs['restarts'].append(now)
    fs.update(strikes=0, frame_hash={}, log_cursor=now, verify_after_restart=True)
    STATE_FILE.write_text(json.dumps(state))
    try:
        subprocess.run([DOCKER, 'restart', FRIGATE_CONTAINER], capture_output=True,
                       text=True, timeout=180, check=True)
        result = 'done'
    except (OSError, subprocess.SubprocessError) as exc:
        result = f'FAILED ({type(exc).__name__})'
        fs['verify_after_restart'] = False
    log(f"[frigate] docker restart {FRIGATE_CONTAINER}: {result}")
    notify('VisionBox: Frigate auto-restarted',
           f'{why}. docker restart: {result}. Health will be re-checked in ~10 min.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='print every probe result')
    parser.add_argument('--dry-run', action='store_true',
                        help='decide but never restart Frigate or send ntfy pushes')
    args = parser.parse_args()
    global DRY_RUN
    DRY_RUN = args.dry_run

    # Hard ceiling: a wedged probe pile-up must not outlive the cron interval.
    # (The lock below already prevents overlapping runs.)
    signal.alarm(280)

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
    restream_up = {}  # camera -> True/False, or missing when the camera itself is down
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
            restream_up[name] = r_ok
            if args.verbose:
                log(f"[{name}] restream={'up' if r_ok else 'DOWN'}")
            st['restream_fails'] = 0 if r_ok else st['restream_fails'] + 1
            if st['restream_fails'] == FAILS_BEFORE_DOWN:
                log(f"[{name}] camera up but go2rtc restream dead — check the frigate container")
                notify('VisionBox: restream problem',
                       f'{name} camera is fine but its go2rtc restream is not answering. '
                       f'Frigate/go2rtc likely needs a restart.')

    try:
        frigate_watchdog(state, restream_up, now, args.verbose)
    except Exception as exc:  # the camera checks above must never be lost to a watchdog bug
        log(f"[frigate] watchdog error: {type(exc).__name__}: {exc}")

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
