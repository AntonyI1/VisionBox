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
  * the API dead (two attempts in a run) while the container runs counts as
    a PIPELINE strike on the wired camera (two consecutive runs -> restart)
  * per-camera symptoms — segments piling up in Frigate's cache ("Too many
    unprocessed recording segments"), latest.jpg frozen on a camera that
    reports frames, a camera whose restream is up but Frigate reads 0 fps, or
    a camera whose restream has no producer — are first ATTRIBUTED (Part 3);
    only PIPELINE-attributed symptoms count as strikes, and a restart needs
    two consecutive strikes (~10 min) on >=2 cameras or on the wired camera
Guards: a container younger than 10 min is left alone; no restart while the
NFS share is unmounted or stale (Frigate would just re-fail — alert instead);
30 min cooldown between auto-restarts; at most 3 per 6 h, then alert-only.

Part 3, attribution before action (added 2026-09-13 after the three Tapo
cameras' WiFi degraded to 30-80 % packet loss and this script restarted Frigate
nine times in one day for "0 fps" / "frozen frame" symptoms that were the
camera link's fault — each restart also cut ~20 s from the healthy wired
camera). Every per-camera symptom is attributed before anything acts on it:
  * CAMERA-LINK when the camera's link is unhealthy (ping loss / RTT), when
    go2rtc's producer for that stream is absent, reconnecting or not receiving
    bytes (sampled twice a few seconds apart via /api/go2rtc/streams), or when
    go2rtc logged read timeouts for that camera's IP since the last run;
  * PIPELINE only when the link and producer are demonstrably healthy and
    Frigate still gets nothing.
  A link problem never restarts Frigate. It raises one "link degraded" push per
  camera after the condition has persisted >=15 min, one "link recovered" push
  after >=15 min clean, and nothing else. "Frigate keeps failing / needs a
  human" is sent at most once per 6 h and only for PIPELINE attribution.
`--dry-run` prints decisions without restarting, rebooting or notifying.
Tunables can be overridden from .env: FRIGATE_API, FRIGATE_CONTAINER,
FRIGATE_NFS_MOUNT, DOCTOR_WARMUP_S, DOCTOR_RESTART_COOLDOWN_S,
DOCTOR_WIRED_CAMERAS, DOCTOR_LINK_LOSS_PCT, DOCTOR_LINK_RTT_MS,
DOCTOR_LINK_HOLD_S, DOCTOR_PRODUCER_SAMPLE_S, DOCTOR_PRODUCER_MIN_BPS,
DOCTOR_PING_COUNT, DOCTOR_HUMAN_REPEAT_S; DOCTOR_STATE_FILE / DOCTOR_ENV_FILE
relocate the state file and .env (used to test a copy against the live box).
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


def _load_env_file(path: Path):
    """python-dotenv when available, else a stdlib KEY=VALUE parser (never overrides)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            if line.startswith('export '):
                line = line[7:]
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except OSError:
        pass


_load_env_file(Path(os.environ.get('DOCTOR_ENV_FILE', ROOT / '.env')))

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

STATE_FILE = Path(os.environ.get('DOCTOR_STATE_FILE', ROOT / 'logs' / 'camera_doctor_state.json'))
FAILS_BEFORE_DOWN = 3
REBOOT_COOLDOWN_S = 3600


def cam_ip(cam: dict) -> str:
    return urlparse(cam['rtsp']).hostname or ''


def log(msg: str):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


RTSP_STATUS_RE = re.compile(rb'^RTSP/1\.0 (\d{3})')


def rtsp_status(url: str, timeout: float = 4.0):
    """Status code of an RTSP DESCRIBE (200, 401, 404, ...) or None when nothing answers."""
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 554),
                                      timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\n"
                          "Accept: application/sdp\r\nUser-Agent: camera-doctor\r\n\r\n").encode())
            m = RTSP_STATUS_RE.match(sock.recv(64))
            return int(m.group(1)) if m else None
    except OSError:
        return None


def rtsp_alive(url: str, timeout: float = 4.0) -> bool:
    """Liveness only: any RTSP status line (401 included) counts as alive."""
    return rtsp_status(url, timeout) is not None


# go2rtc answers DESCRIBE with "404 Not Found" when it has no producer for the
# stream (measured 2026-09-13); that is NOT a usable restream.
RESTREAM_OK_STATUS = (200, 401)


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
    if DRY_RUN:
        log(f"[dry-run] would push: {title}: {message}")
        return
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

# --- attribution (Part 3) ---
WIRED_CAMERAS = tuple(c for c in os.environ.get('DOCTOR_WIRED_CAMERAS', 'front_door').split(',') if c)
LINK_LOSS_PCT = float(os.environ.get('DOCTOR_LINK_LOSS_PCT', 20))    # >= this ping loss = link unhealthy
LINK_RTT_MS = float(os.environ.get('DOCTOR_LINK_RTT_MS', 150))       # >= this avg RTT (LAN!) = link unhealthy
LINK_HOLD_S = int(os.environ.get('DOCTOR_LINK_HOLD_S', 900))         # hysteresis for link pushes
LINK_HOLD_SLACK_S = 30                                               # cron jitter tolerance
PRODUCER_SAMPLE_S = float(os.environ.get('DOCTOR_PRODUCER_SAMPLE_S', 6))
PRODUCER_MIN_BPS = float(os.environ.get('DOCTOR_PRODUCER_MIN_BPS', 1000))
PING = '/usr/bin/ping'
PING_COUNT = int(os.environ.get('DOCTOR_PING_COUNT', 10))
HUMAN_REPEAT_S = int(os.environ.get('DOCTOR_HUMAN_REPEAT_S', 6 * 3600))
LINK_ALERT_REPEAT_S = int(os.environ.get('DOCTOR_LINK_ALERT_REPEAT_S', 6 * 3600))  # per camera
PIPELINE_STRIKES = 2        # consecutive runs of PIPELINE symptoms before a restart
CAPTURE_RSS_ALERT_MB = float(os.environ.get('DOCTOR_CAPTURE_RSS_ALERT_MB', 1500))  # frigate.capture balloon warning
HUMAN_AFTER_STRIKES = 3     # runs of an unrestartable PIPELINE symptom before "needs a human"

TS_RE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
LOG_SIGNATURES = {
    'dead_thread': re.compile(r'Exception in thread (\S+?):'),
    'unprocessed': re.compile(r'Too many unprocessed recording segments in cache for (\w+)'),
    'estale': re.compile(r'Stale file handle'),
    'va_error': re.compile(r'No VA display found|Hardware device setup failed'),
}
GO2RTC_LINK_RE = re.compile(r'i/o timeout|RTP header size insufficient|connection reset|'
                            r'connection refused|no route to host|unexpected EOF|\bEOF\b')
GO2RTC_IP_RE = re.compile(r'(\d{1,3}(?:\.\d{1,3}){3}):554')
PING_LOSS_RE = re.compile(r'(\d+) packets transmitted, (\d+) received')
PING_RTT_RE = re.compile(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/')


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


def parse_go2rtc_log(lines, since: float, ip_to_cam: dict) -> dict:
    """Pure. Count go2rtc producer link errors per camera in lines newer than `since`.

    {'timeouts': {camera: n}, 'newest': epoch}. Lines carry the same UTC prefix
    as the Frigate log; the camera is identified by its IP (":554") in the line.
    """
    found = {'timeouts': {}, 'newest': since}
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
        if not GO2RTC_LINK_RE.search(line):
            continue
        for ip in GO2RTC_IP_RE.findall(line):
            cam = ip_to_cam.get(ip)
            if cam:
                found['timeouts'][cam] = found['timeouts'].get(cam, 0) + 1
                break
    return found


def parse_ping(text: str):
    """Pure. {'sent','received','loss_pct','rtt_avg_ms','rtt_max_ms'} or None if unparseable."""
    m = PING_LOSS_RE.search(text or '')
    if not m:
        return None
    sent, received = int(m.group(1)), int(m.group(2))
    res = {'sent': sent, 'received': received,
           'loss_pct': 100.0 * (sent - received) / sent if sent else 100.0,
           'rtt_avg_ms': None, 'rtt_max_ms': None}
    r = PING_RTT_RE.search(text)
    if r:
        res['rtt_avg_ms'] = float(r.group(2))
        res['rtt_max_ms'] = float(r.group(3))
    return res


def start_ping(ip: str):
    try:
        return subprocess.Popen([PING, '-n', '-q', '-c', str(PING_COUNT), '-i', '0.2', '-W', '1', ip],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError:
        return None


def finish_ping(proc, timeout: float = 15):
    if proc is None:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None
    return parse_ping(out)


def fetch_streams():
    """go2rtc's stream table through Frigate's proxy, or None when unavailable."""
    try:
        data = json.loads(http_get(f'{FRIGATE_API}/api/go2rtc/streams'))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def producer_snapshot(streams):
    """Pure. stream name -> (producer id, bytes received) or None when the stream has no producer."""
    if streams is None:
        return None
    snap = {}
    for name, s in streams.items():
        prods = (s or {}).get('producers') or []
        if prods:
            p = prods[0]
            snap[name] = (p.get('id'), int(p.get('bytes_recv', p.get('recv', 0)) or 0))
        else:
            snap[name] = None
    return snap


def producer_health(snap1, snap2, stream: str, interval_s: float,
                    min_bps: float = PRODUCER_MIN_BPS) -> dict:
    """Pure. Is go2rtc's producer for `stream` present, stable and receiving bytes?"""
    res = {'sampled': False, 'present': False, 'reconnected': False,
           'bytes': None, 'bps': None, 'starved': False, 'id': None}
    if snap1 is None or snap2 is None:
        return res
    res['sampled'] = True
    a, b = snap1.get(stream), snap2.get(stream)
    if a is None or b is None:
        res['reconnected'] = (a is None) != (b is None)
        res['starved'] = True
        res['id'] = (b or a or (None,))[0]
        return res
    res['present'] = True
    res['id'] = b[0]
    if a[0] != b[0]:
        res['reconnected'] = True
        res['starved'] = True
        return res
    res['bytes'] = max(0, b[1] - a[1])
    res['bps'] = res['bytes'] / interval_s if interval_s > 0 else 0.0
    res['starved'] = res['bps'] < min_bps
    return res


def link_health(cam_alive: bool, ping, producer: dict, go2rtc_timeouts: int,
                loss_pct: float = LINK_LOSS_PCT, rtt_ms: float = LINK_RTT_MS) -> dict:
    """Pure. {'degraded': bool, 'measured': bool, 'reasons': [..]} for one camera's link."""
    reasons = []
    measured = ping is not None or bool(producer.get('sampled'))
    if not cam_alive:
        reasons.append('camera RTSP not answering')
    if ping is not None:
        if ping['loss_pct'] >= loss_pct:
            reasons.append(f"ping loss {ping['loss_pct']:.0f}%")
        if ping['rtt_avg_ms'] is not None and ping['rtt_avg_ms'] >= rtt_ms:
            reasons.append(f"ping avg RTT {ping['rtt_avg_ms']:.0f} ms")
    if producer.get('sampled'):
        if not producer['present']:
            reasons.append('go2rtc has no producer for the stream')
        elif producer['reconnected']:
            reasons.append('go2rtc producer reconnected during sampling')
        elif producer['starved']:
            reasons.append(f"go2rtc producer starved ({producer['bps']:.0f} B/s)")
    if go2rtc_timeouts:
        reasons.append(f'{go2rtc_timeouts} go2rtc read timeouts since last run')
    return {'degraded': bool(reasons), 'measured': measured, 'reasons': reasons}


def reboot_allowed(last: dict, now: float, loss_pct: float = LINK_LOSS_PCT,
                   rtt_ms: float = LINK_RTT_MS) -> tuple[bool, str]:
    """Pure. May a camera that fails its RTSP probe be sent an ONVIF reboot?

    Yes when nothing fresh is known about its link (old behaviour) or when it
    answers ping cleanly (the hung-firmware signature: IP stack alive, RTSP
    dead). No when it does not answer ping at all (the reboot command could not
    arrive either) or when the link is lossy/slow (a radio problem, not a hang -
    front_garage was rebooted for nothing on 2026-09-13 02:00 UTC).
    """
    if not last or last.get('at') != now or last.get('loss_pct') is None:
        return True, ''
    loss, rtt = last['loss_pct'], last.get('rtt_avg_ms')
    if loss >= 100:
        return False, 'no ping reply at all - unreachable, an ONVIF reboot command could not arrive either'
    if loss >= loss_pct or (rtt is not None and rtt >= rtt_ms):
        return False, (f'camera answers ping but its link is lossy/slow (loss {loss:.0f}%, avg RTT '
                       f'{rtt or 0:.0f} ms) - a reboot cannot fix a radio problem')
    return True, ''


def dead_processes(cams_stats: dict, alive: dict, enabled) -> dict:
    """Pure. camera -> dead roles among Frigate's own per-camera processes.

    /api/stats reports `capture_pid` (frigate.capture) and `pid` (the tracker).
    Frigate never restarts these when they die (e.g. the container's memory cap
    OOM-killed frigate.capture:front_garage on 2026-09-14 00:26 UTC after a
    post-stall frame burst), so a dead one is a PIPELINE fault whatever the
    camera link does. `alive` maps pid -> bool; unknown pids are not flagged.
    """
    out = {}
    for cam in enabled:
        st = cams_stats.get(cam) or {}
        dead = []
        for role, key in (('capture', 'capture_pid'), ('tracker', 'pid')):
            pid = st.get(key)
            try:
                pid = int(pid) if pid else 0
            except (TypeError, ValueError):
                pid = 0
            if pid and alive.get(pid) is False:
                dead.append(role)
        if dead:
            out[cam] = dead
    return out


def probe_pids(pids) -> dict:
    """pid -> alive? inside the Frigate container; {} when it cannot be determined."""
    pids = sorted({int(p) for p in pids if p})
    if not pids:
        return {}
    script = 'for p in ' + ' '.join(str(p) for p in pids) + '; do if [ -d /proc/$p ]; then echo $p 1; else echo $p 0; fi; done'
    try:
        out = subprocess.run([DOCKER, 'exec', FRIGATE_CONTAINER, 'sh', '-c', script],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    res = {}
    for line in out.stdout.split('\n'):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in ('0', '1'):
            res[int(parts[0])] = parts[1] == '1'
    return res


def probe_rss(pids) -> dict:
    """pid -> resident MB inside the Frigate container; {} when unavailable."""
    pids = sorted({int(p) for p in pids if p})
    if not pids:
        return {}
    try:
        out = subprocess.run([DOCKER, 'exec', FRIGATE_CONTAINER, 'ps', '-o', 'pid=,rss=',
                              '-p', ','.join(str(p) for p in pids)],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.stdout.split('\n'):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            res[int(parts[0])] = int(parts[1]) / 1024.0
    return res


def attribute(cam_symptoms: dict, link: dict) -> dict:
    """Pure. camera -> 'link' | 'pipeline' | 'unknown' for every camera with a symptom.

    A symptom is the LINK's fault whenever the link is unhealthy or go2rtc's
    producer is flapping/starved; it is the PIPELINE's fault only when the link
    was measured healthy and Frigate still gets nothing; 'unknown' when nothing
    could be measured (never restarts).
    """
    out = {}
    for cam in cam_symptoms:
        lk = link.get(cam) or {}
        if lk.get('degraded'):
            out[cam] = 'link'
        elif lk.get('measured'):
            out[cam] = 'pipeline'
        else:
            out[cam] = 'unknown'
    return out


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


def decide(immediate: dict, attribution: dict, strikes: int,
           wired=WIRED_CAMERAS, solo=()) -> tuple[bool, int]:
    """Pure policy: (restart?, new strike count).

    Only PIPELINE-attributed symptoms count. A restart needs an immediate fault,
    or PIPELINE_STRIKES consecutive runs of pipeline symptoms that hit >=2
    cameras, a wired camera, or a camera in `solo` (its own Frigate process is
    dead - unambiguous, and Frigate cannot recover it). Otherwise a single WiFi
    camera can never restart Frigate.
    """
    pipeline = [c for c, a in attribution.items() if a == 'pipeline']
    strikes = strikes + 1 if pipeline else 0
    broad = (len(pipeline) >= 2 or any(c in wired for c in pipeline)
             or any(c in solo for c in pipeline))
    restart = bool(immediate) or (broad and strikes >= PIPELINE_STRIKES)
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


def link_transition(ls: dict, degraded: bool, now: float,
                    hold_s: float = LINK_HOLD_S) -> tuple[dict, str]:
    """Pure hysteresis: returns (new link state, event) with event in
    ('', 'degraded', 'recovered'). 'degraded' fires once the condition has held
    >= hold_s; 'recovered' once it has been clean >= hold_s after a 'degraded'."""
    ls = dict(ls)
    event = ''
    hold = hold_s - LINK_HOLD_SLACK_S
    if degraded:
        ls['clean_since'] = None
        if ls.get('bad_since') is None:
            ls['bad_since'] = now
        if ls.get('status') != 'degraded' and now - ls['bad_since'] >= hold:
            ls['status'] = 'degraded'
            event = 'degraded'
    else:
        ls['bad_since'] = None
        if ls.get('clean_since') is None:
            ls['clean_since'] = now
        if ls.get('status') == 'degraded' and now - ls['clean_since'] >= hold:
            ls['status'] = 'ok'
            event = 'recovered'
    return ls, event


def alert_due(alerts: dict, key: str, now: float, repeat: float = ALERT_REPEAT_S) -> bool:
    """Pure dedupe: may `key` be pushed again at `now`?"""
    return now - alerts.get(key, 0) >= repeat


def alert(state: dict, key: str, title: str, message: str, now: float,
          repeat: float = ALERT_REPEAT_S, tag: str = '[frigate]') -> bool:
    """Deduped push; returns True when it was actually sent (or would be, in dry-run)."""
    alerts = state.setdefault('frigate', {}).setdefault('alerts', {})
    if not alert_due(alerts, key, now, repeat):
        return False
    alerts[key] = now
    log(f"{tag} ALERT {title}: {message}")
    notify(title, message)
    return True


CAM_STATE_DEFAULTS = {'fails': 0, 'down_since': None, 'last_reboot': 0,
                      'notified': False, 'restream_fails': 0, 'reboot_skipped_at': 0}
LINK_STATE_DEFAULTS = {'status': 'ok', 'bad_since': None, 'clean_since': None,
                       'alerted': False, 'last': {}}
FRIGATE_STATE_DEFAULTS = {'strikes': 0, 'restarts': [], 'frame_hash': {},
                          'alerts': {}, 'down_runs': 0, 'attribution': {}}


def normalise_state(state: dict) -> dict:
    """Fill in defaults for older state files. Adds keys, never renames or drops any."""
    if not isinstance(state, dict):
        state = {}
    for name in CAMERAS:
        st = state.setdefault(name, {})
        for k, v in CAM_STATE_DEFAULTS.items():
            st.setdefault(k, v)
        ls = st.setdefault('link', {})
        for k, v in LINK_STATE_DEFAULTS.items():
            ls.setdefault(k, dict(v) if isinstance(v, dict) else v)
    fs = state.setdefault('frigate', {})
    for k, v in FRIGATE_STATE_DEFAULTS.items():
        fs.setdefault(k, type(v)() if isinstance(v, (dict, list)) else v)
    # the old 'budget' alert becomes the 6-h 'needs_human' one; carry its timestamp over
    fs['alerts'].setdefault('needs_human', fs['alerts'].get('budget', 0))
    return state


def measure_links(state: dict, alive: dict, now: float, verbose: bool) -> dict:
    """Ping every camera, sample go2rtc's producers twice and count go2rtc read
    timeouts since the last run. Returns camera -> link_health() dict and stores
    the measurements in state[cam]['link']['last']."""
    fs = state['frigate']
    ip_to_cam = {cam_ip(c): n for n, c in CAMERAS.items()}
    pings = {n: start_ping(cam_ip(c)) for n, c in CAMERAS.items()}
    t0 = time.time()
    snap1 = producer_snapshot(fetch_streams())
    time.sleep(PRODUCER_SAMPLE_S)
    snap2 = producer_snapshot(fetch_streams()) if snap1 is not None else None
    interval = time.time() - t0
    ping_res = {n: finish_ping(p) for n, p in pings.items()}

    timeouts = {}
    cursor = fs.get('go2rtc_cursor')
    since = now - LOG_WINDOW_S if cursor is None else max(cursor, now - 3600)
    try:
        payload = json.loads(http_get(f'{FRIGATE_API}/api/logs/go2rtc?start=-400', timeout=15))
        g = parse_go2rtc_log(payload.get('lines', []), since, ip_to_cam)
        timeouts = g['timeouts']
        fs['go2rtc_cursor'] = g['newest']
    except Exception:
        pass

    link = {}
    for name, cam in CAMERAS.items():
        producer = producer_health(snap1, snap2, cam['restream'], interval)
        ping = ping_res.get(name)
        lh = link_health(alive.get(name, False), ping, producer, timeouts.get(name, 0))
        link[name] = lh
        state[name]['link']['last'] = {
            'at': now, 'loss_pct': ping['loss_pct'] if ping else None,
            'rtt_avg_ms': ping['rtt_avg_ms'] if ping else None,
            'rtt_max_ms': ping['rtt_max_ms'] if ping else None,
            'producer_bps': producer['bps'], 'producer_id': producer['id'],
            'producer_present': producer['present'], 'producer_sampled': producer['sampled'],
            'go2rtc_timeouts': timeouts.get(name, 0), 'degraded': lh['degraded'],
            'reasons': lh['reasons'],
        }
        if verbose:
            ptxt = (f"loss={ping['loss_pct']:.0f}% avg={ping['rtt_avg_ms']} ms max={ping['rtt_max_ms']} ms"
                    if ping else 'ping=n/a')
            if producer['sampled']:
                btxt = (f"producer={'present' if producer['present'] else 'MISSING'}"
                        f"{' RECONNECTED' if producer['reconnected'] else ''}"
                        + (f" {producer['bps'] / 1000:.1f} kB/s" if producer['bps'] is not None else ''))
            else:
                btxt = 'producer=unsampled'
            log(f"[link] {name}: {ptxt} | {btxt} | go2rtc_timeouts={timeouts.get(name, 0)} -> "
                f"{'DEGRADED (' + '; '.join(lh['reasons']) + ')' if lh['degraded'] else 'ok'}")
    return link


def link_message(name: str, st: dict, now: float) -> str:
    last = st['link'].get('last', {})
    ip = cam_ip(CAMERAS[name])
    since = st['link'].get('bad_since') or now
    parts = [f"{name} ({ip}) link degraded for {(now - since) / 60:.0f} min:"]
    if last.get('loss_pct') is not None:
        parts.append(f"{last['loss_pct']:.0f}% ping loss,")
    if last.get('rtt_avg_ms') is not None:
        parts.append(f"avg RTT {last['rtt_avg_ms']:.0f} ms (max {last.get('rtt_max_ms') or 0:.0f} ms),")
    if last.get('producer_sampled'):
        if not last.get('producer_present'):
            parts.append('go2rtc has no stream from it,')
        elif last.get('producer_bps') is not None:
            parts.append(f"go2rtc receiving {last['producer_bps'] / 1000:.0f} kB/s,")
    parts.append(f"{last.get('go2rtc_timeouts', 0)} go2rtc read timeouts in the last 5 min.")
    parts.append('Frigate/go2rtc are reconnecting on their own and record whatever arrives; '
                 'nothing on the server needs a restart.')
    parts.append("Do: check this camera's WiFi (signal, 2.4 GHz interference, the AP), "
                 'or power-cycle the camera if it stays like this; no server action.')
    return ' '.join(parts)


def frigate_watchdog(state: dict, restream_up: dict, now: float, verbose: bool, link: dict = None):
    fs = state.setdefault('frigate', {})
    fs.setdefault('strikes', 0)
    fs.setdefault('restarts', [])
    fs.setdefault('frame_hash', {})
    fs.setdefault('alerts', {})
    fs.setdefault('down_runs', 0)
    link = link or {}

    status, started = container_state()
    if status != 'running':
        fs['down_runs'] += 1
        fs['strikes'] = 0
        if fs['down_runs'] >= 2:
            alert(state, 'not_running', 'VisionBox: Frigate container not running',
                  f"docker reports the frigate container is '{status}'. Docker's restart "
                  f"policy should bring it back. Do: run `docker ps -a` and "
                  f"`docker start frigate` if it stays down.", now)
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
        alert(state, 'nfs', 'VisionBox: NFS share problem',
              '; '.join(nfs_problems) + '. Frigate cannot record until the NAS export is '
              'healthy; not restarting it. Do: check the NAS / remount the export.', now)

    immediate, symptoms, cam_symptoms = {}, {}, {}
    dead = {}

    def cam_symptom(cam: str, key: str):
        cam_symptoms.setdefault(cam, []).append(key)

    cams = None
    for attempt in range(2):
        try:
            cams = json.loads(http_get(f'{FRIGATE_API}/api/stats', timeout=10 + 5 * attempt)).get('cameras', {})
            break
        except Exception as exc:
            api_err = f'/api/stats unreachable twice ({type(exc).__name__}) while the container runs'
            if attempt == 0:
                time.sleep(5)
    if cams is None:
        # A dead API is a PIPELINE symptom whatever the camera links do, but it is a strike, not an
        # immediate restart: two consecutive runs (~10 min) must see it, as in the original script.
        symptoms['api_down'] = api_err
        for c in WIRED_CAMERAS:
            cam_symptom(c, 'api_down')

    cursor = fs.get('log_cursor')
    since = now - LOG_WINDOW_S if cursor is None else max(cursor, now - 3600)
    try:
        payload = json.loads(http_get(f'{FRIGATE_API}/api/logs/frigate?start=-1500', timeout=15))
        found = parse_frigate_log(payload.get('lines', []), since)
        fs['log_cursor'] = found['newest']
    except Exception as exc:
        found = None
        log(f'[frigate] /api/logs unreachable ({type(exc).__name__}); log signatures skipped this run')

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
            for c in found['unprocessed']:
                cam_symptom(c, 'unprocessed')
        if found['estale']:
            alert(state, 'estale', 'VisionBox: NFS stale file handles',
                  f"Frigate logged {found['estale']} 'Stale file handle' errors — the NAS "
                  f"export changed underneath it. Watching for a dead thread. Do: nothing yet; "
                  f"an auto-restart follows if a thread dies.", now)

    if cams is not None:
        try:
            cfg_cams = json.loads(http_get(f'{FRIGATE_API}/api/config')).get('cameras', {})
        except Exception:
            cfg_cams = {}
        enabled = [c for c in CAMERAS if cfg_cams.get(c, {}).get('enabled', True)]
        fps = {c: (cams.get(c) or {}).get('camera_fps') or 0 for c in enabled}
        # Frigate's own per-camera processes: dead = pipeline fault, Frigate never restarts them.
        pid_list = [p for c in enabled for p in ((cams.get(c) or {}).get('capture_pid'), (cams.get(c) or {}).get('pid')) if p]
        dead = dead_processes(cams, probe_pids(pid_list), enabled)
        if dead:
            symptoms['process_dead'] = ('Frigate camera process dead (Frigate does not restart it): '
                                        + ', '.join(f"{c} ({'+'.join(r)})" for c, r in dead.items()))
            for c in dead:
                cam_symptom(c, 'process_dead')
        # early warning: frigate.capture ballooning towards the container's memory cap
        rss = probe_rss([(cams.get(c) or {}).get('capture_pid') for c in enabled])
        for c in enabled:
            cp = (cams.get(c) or {}).get('capture_pid')
            mb = rss.get(int(cp)) if cp else None
            if mb and mb >= CAPTURE_RSS_ALERT_MB:
                alert(state, f'rss_{c}', f'VisionBox: {c} capture process ballooning',
                      f"frigate.capture for {c} uses {mb:.0f} MB (container cap 4 GB). After a WiFi "
                      f"stall this process can balloon and be OOM-killed, which silently stops the "
                      f"camera. Do: nothing now; the watchdog restarts Frigate if the process dies.",
                      now, repeat=HUMAN_REPEAT_S)
        zero = [c for c in enabled if restream_up.get(c) and fps[c] == 0]
        if zero:
            symptoms['zero_fps'] = 'restream up but Frigate reads 0 fps: ' + ', '.join(zero)
            for c in zero:
                cam_symptom(c, 'zero_fps')
        # camera answers directly but go2rtc has no usable restream for it (404 / no answer)
        no_restream = [c for c in enabled if restream_up.get(c) is False]
        if no_restream:
            symptoms['no_restream'] = 'camera up but go2rtc restream not serving: ' + ', '.join(no_restream)
            for c in no_restream:
                cam_symptom(c, 'no_restream')
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
            for c in frozen:
                cam_symptom(c, 'frozen')

    attribution = attribute(cam_symptoms, link)
    if 'api_down' in symptoms:
        for c in WIRED_CAMERAS:
            attribution[c] = 'pipeline'
    for c in dead:  # a dead Frigate process is never the link's fault
        attribution[c] = 'pipeline'
    fs['attribution'] = attribution
    restart, fs['strikes'] = decide(immediate, attribution, fs['strikes'], solo=tuple(dead))
    pipeline_cams = sorted(c for c, a in attribution.items() if a == 'pipeline')
    link_cams = sorted(c for c, a in attribution.items() if a != 'pipeline')

    def describe(cams_):
        return '; '.join(f"{c}: {'+'.join(cam_symptoms[c])}"
                         + (f" [{', '.join(link[c]['reasons'])}]" if link.get(c, {}).get('reasons') else '')
                         for c in cams_)

    if verbose:
        log(f"[frigate] immediate={immediate or '-'} symptoms={symptoms or '-'} "
            f"attribution={attribution or '-'} strikes={fs['strikes']} restart={restart}")
    if not restart:
        if link_cams:
            log(f"[frigate] link-attributed, no restart: {describe(link_cams)}")
        if pipeline_cams:
            log(f"[frigate] strike {fs['strikes']} (pipeline): {describe(pipeline_cams)}")
            if fs['strikes'] >= HUMAN_AFTER_STRIKES:
                alert(state, 'needs_human', 'VisionBox: Frigate keeps failing',
                      f"Pipeline fault for {fs['strikes'] * 5} min on {', '.join(pipeline_cams)} "
                      f"({describe(pipeline_cams)}); a single WiFi camera does not trigger an "
                      f"auto-restart. Do: open Frigate's System/Logs page; run "
                      f"`docker restart frigate` yourself if its recordings are missing.",
                      now, repeat=HUMAN_REPEAT_S)
        elif fs.pop('verify_after_restart', False):
            note = (f" Link still degraded on {', '.join(link_cams)} (camera side; Frigate "
                    f"reconnects on its own)." if link_cams else '')
            log('[frigate] healthy after auto-restart')
            notify('VisionBox: Frigate healthy again',
                   f'Pipeline checks are clean after the auto-restart.{note} Do: nothing.')
        return

    why = '; '.join(list(immediate.values()) + [describe(pipeline_cams)] if pipeline_cams
                    else list(immediate.values()))
    blocked = restart_blocked(fs, now, nfs_problems)
    if blocked:
        log(f"[frigate] restart wanted ({why}) but not doing it: {blocked}")
        alert(state, 'needs_human', 'VisionBox: Frigate keeps failing',
              f'{why}. Auto-restart blocked: {blocked}; needs a human. Do: check Frigate\'s '
              f'System/Logs page and the NAS, then `docker restart frigate` yourself.',
              now, repeat=HUMAN_REPEAT_S)
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
           f'{why}. docker restart: {result}. Health will be re-checked in ~10 min. '
           f"Do: nothing unless a 'keeps failing' push follows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='print every probe result')
    parser.add_argument('--dry-run', action='store_true',
                        help='decide but never restart Frigate, reboot a camera or send ntfy pushes')
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
    state = normalise_state(state)

    now = time.time()
    restream_up = {}  # camera -> True/False, or missing when the camera itself is down
    alive_map = {}
    reboot_wanted = []
    for name, cam in CAMERAS.items():
        st = state[name]
        alive = rtsp_alive(cam['rtsp'])
        alive_map[name] = alive
        if args.verbose:
            log(f"[{name}] camera={'up' if alive else 'DOWN'}")

        if alive:
            if st['down_since']:
                mins = (now - st['down_since']) / 60
                log(f"[{name}] RECOVERED after {mins:.0f} min")
                if st['notified']:
                    notify(f'VisionBox: {name} recovered',
                           f'{name} is answering again after {mins:.0f} min down. Do: nothing.')
            st.update(fails=0, down_since=None, notified=False)
        else:
            st['fails'] += 1
            if st['fails'] == FAILS_BEFORE_DOWN:
                st['down_since'] = now - FAILS_BEFORE_DOWN * 300
                log(f"[{name}] DOWN ({FAILS_BEFORE_DOWN} consecutive probe failures)")
                notify(f'VisionBox: {name} DOWN',
                       f'{name} RTSP has not answered for ~15 min. '
                       f'Attempting ONVIF reboot. Do: if it stays down, power-cycle the camera.')
                st['notified'] = True
            if st['fails'] >= FAILS_BEFORE_DOWN and now - st['last_reboot'] > REBOOT_COOLDOWN_S:
                reboot_wanted.append(name)  # decided after the link measurement below

        # Restream health only means anything while the camera itself is up —
        # otherwise it's the same outage seen through go2rtc.
        if alive:
            r_status = rtsp_status(f"rtsp://127.0.0.1:8554/{cam['restream']}")
            r_ok = r_status in RESTREAM_OK_STATUS
            restream_up[name] = r_ok
            if args.verbose:
                log(f"[{name}] restream={'up' if r_ok else 'DOWN'} (status {r_status})")
            st['restream_fails'] = 0 if r_ok else st['restream_fails'] + 1

    # Link measurements (ping, go2rtc producer bytes, go2rtc read timeouts) and
    # the per-camera link alerts with 15-min hysteresis.
    link = {}
    try:
        link = measure_links(state, alive_map, now, args.verbose)
        for name in CAMERAS:
            st = state[name]
            ls, event = link_transition(st['link'], link[name]['degraded'], now)
            ls.setdefault('alerted', False)
            if event == 'degraded':
                log(f"[link] {name} DEGRADED for >= {LINK_HOLD_S // 60} min: "
                    + '; '.join(link[name]['reasons']))
                if not st['down_since']:  # Part 1 already pushed "DOWN" for a dead camera
                    st['link'] = ls
                    ls['alerted'] = alert(state, f'link_{name}', f'VisionBox: {name} link degraded',
                                          link_message(name, st, now), now,
                                          repeat=LINK_ALERT_REPEAT_S, tag='[link]')
            elif event == 'recovered':
                last = ls.get('last', {})
                log(f"[link] {name} link recovered (clean for >= {LINK_HOLD_S // 60} min)")
                if ls.get('alerted'):
                    notify(f'VisionBox: {name} link recovered',
                           f"{name} link has been clean for {LINK_HOLD_S // 60} min "
                           f"({last.get('loss_pct') or 0:.0f}% loss, avg RTT "
                           f"{last.get('rtt_avg_ms') or 0:.0f} ms). Do: nothing.")
                ls['alerted'] = False
            st['link'] = ls
            # "restream problem" (3 runs without a usable restream) is only
            # Frigate's fault when the camera's link is clean.
            if st['restream_fails'] == FAILS_BEFORE_DOWN:
                if link[name]['degraded']:
                    log(f"[{name}] camera up but go2rtc restream not serving — link degraded "
                        f"({'; '.join(link[name]['reasons'])}), not blaming Frigate")
                else:
                    log(f"[{name}] camera up but go2rtc restream dead — check the frigate container")
                    alert(state, f'restream_{name}', 'VisionBox: restream problem',
                          f'{name} camera and its link are fine but its go2rtc restream is not '
                          f'answering. Do: `docker restart frigate` if this repeats.', now, tag=f'[{name}]')
    except Exception as exc:  # link measurement must never break the camera checks
        log(f"[link] error: {type(exc).__name__}: {exc}")

    # Deferred camera reboots (Part 1): only a camera that is dead over RTSP but
    # answers ping cleanly looks like hung firmware; a lossy radio link is not
    # fixed by rebooting the camera.
    for name in reboot_wanted:
        st = state[name]
        ok, why = reboot_allowed(st['link'].get('last') or {}, now)
        if not ok:
            if now - st.get('reboot_skipped_at', 0) > REBOOT_COOLDOWN_S:
                st['reboot_skipped_at'] = now
                log(f"[{name}] not sending ONVIF reboot: {why}")
            continue
        if DRY_RUN:
            log(f"[{name}] dry-run: would send ONVIF reboot")
            continue
        st['last_reboot'] = now
        result = onvif_reboot(*CAMERAS[name]['onvif'])
        log(f"[{name}] ONVIF reboot attempt: {result}")

    try:
        frigate_watchdog(state, restream_up, now, args.verbose, link)
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
