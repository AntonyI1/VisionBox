#!/usr/bin/env python3
"""Camera and Frigate watchdog, run from cron every five minutes.

Three layers, each acting only on what the one before could not explain:

Cameras. Each camera's own RTSP endpoint and its go2rtc restream are probed
for liveness (any RTSP status line, 401 included, counts as alive). A camera
that fails FAILS_BEFORE_DOWN consecutive runs is declared DOWN, pushed via
ntfy and sent an ONVIF SystemReboot at most once an hour, which heals a hung
RTSP daemon. A camera that is unreachable or on a lossy link is not rebooted:
the command could not arrive, or the radio is the problem.

Frigate pipeline. Only Frigate-internal liveness signals are watched, never
motion or recording activity, so a quiet camera cannot trigger anything. A
dead thread or VAAPI device errors in the Frigate log restart the container
at once. Everything else is a strike: the API dead twice in one run, recording
segments piling up in the cache, latest.jpg frozen on a camera that reports
frames, a restream that is up while Frigate reads 0 fps, a restream without a
producer, or a dead frigate.capture/tracker process (Frigate never restarts
those). A restart needs PIPELINE_STRIKES consecutive strikes on two or more
cameras, on a wired camera, or on a camera whose own Frigate process died. A
container younger than WARMUP_S is left alone, nothing is restarted while the
NFS share is unmounted or stale, restarts are RESTART_COOLDOWN_S apart and
capped at RESTART_BUDGET_N per window; beyond that only a "needs a human" push
goes out.

Attribution. Before a per-camera symptom counts as a strike its link is
measured: ping loss and RTT, go2rtc's producer for the stream (present, stable
and receiving bytes across two samples) and go2rtc read timeouts logged for
the camera's IP since the last run. A symptom on an unhealthy link is the
link's fault and never restarts Frigate; it raises one "link degraded" push
per camera once the condition has held LINK_HOLD_S and one "link recovered"
push after as long clean. Only a symptom on a demonstrably healthy link is a
pipeline strike.

Policy helpers are pure functions and are unit-tested directly. --dry-run
prints every decision without restarting, rebooting or notifying. Tunables
can be overridden from .env: FRIGATE_API, FRIGATE_CONTAINER,
FRIGATE_NFS_MOUNT, DOCTOR_WARMUP_S, DOCTOR_RESTART_COOLDOWN_S,
DOCTOR_WIRED_CAMERAS, DOCTOR_LINK_LOSS_PCT, DOCTOR_LINK_RTT_MS,
DOCTOR_LINK_HOLD_S, DOCTOR_LINK_ALERT_REPEAT_S, DOCTOR_PRODUCER_SAMPLE_S,
DOCTOR_PRODUCER_MIN_BPS, DOCTOR_PING_COUNT, DOCTOR_HUMAN_REPEAT_S,
DOCTOR_CAPTURE_RSS_ALERT_MB, NTFY_TOPIC, NTFY_URL; DOCTOR_STATE_FILE and
DOCTOR_ENV_FILE relocate the state file and .env so a copy can be exercised
against the live box.
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
import threading
import time
import urllib.request
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path):
    """python-dotenv when available, else a stdlib KEY=VALUE parser; neither overrides the environment."""
    if load_dotenv is not None:
        load_dotenv(path)
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.removeprefix('export ').partition('=')
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

# rtsp: the camera's own endpoint, which isolates camera health from go2rtc/Frigate;
# restream: the go2rtc stream Frigate and VisionBox consume; onvif: the SystemReboot target.
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
LOG_FILE = ROOT / 'logs' / 'camera_doctor.log'
DOCKER = '/usr/bin/docker'
PING = '/usr/bin/ping'

# Cameras
FAILS_BEFORE_DOWN = 3            # consecutive failed runs (~15 min) before a camera is DOWN
REBOOT_COOLDOWN_S = 3600
RESTREAM_OK_STATUS = (200, 401)  # go2rtc answers 404 when it has no producer: not a usable restream

# Frigate pipeline
FRIGATE_API = os.environ.get('FRIGATE_API', 'http://127.0.0.1:5000').rstrip('/')
FRIGATE_CONTAINER = os.environ.get('FRIGATE_CONTAINER', 'frigate')
NFS_MOUNT = Path(os.environ.get('FRIGATE_NFS_MOUNT', '/mnt/storage'))
NFS_PROBE_DIRS = ('frigate/clips', 'frigate/recordings')
WARMUP_S = int(os.environ.get('DOCTOR_WARMUP_S', '600'))
RESTART_COOLDOWN_S = int(os.environ.get('DOCTOR_RESTART_COOLDOWN_S', '1800'))
RESTART_BUDGET_N, RESTART_BUDGET_WINDOW_S = 3, 6 * 3600
PIPELINE_STRIKES = 2             # consecutive runs of pipeline symptoms before a restart
HUMAN_AFTER_STRIKES = 3          # runs of an unrestartable pipeline symptom before "needs a human"
UNPROCESSED_SUSTAINED = 6        # maintainer warnings per run that mean "stuck", not a blip
LOG_WINDOW_S = 6 * 60            # first-run lookback; later runs continue from a cursor
ALERT_REPEAT_S = 3600
HUMAN_REPEAT_S = int(os.environ.get('DOCTOR_HUMAN_REPEAT_S', str(6 * 3600)))
CAPTURE_RSS_ALERT_MB = float(os.environ.get('DOCTOR_CAPTURE_RSS_ALERT_MB', '1500'))

# Link attribution
WIRED_CAMERAS = tuple(c for c in os.environ.get('DOCTOR_WIRED_CAMERAS', 'front_door').split(',') if c)
LINK_LOSS_PCT = float(os.environ.get('DOCTOR_LINK_LOSS_PCT', '20'))  # ping loss from which a link is unhealthy
LINK_RTT_MS = float(os.environ.get('DOCTOR_LINK_RTT_MS', '150'))     # average LAN RTT from which it is unhealthy
LINK_HOLD_S = int(os.environ.get('DOCTOR_LINK_HOLD_S', '900'))       # hysteresis for the link pushes
LINK_HOLD_SLACK_S = 30                                               # cron jitter tolerance
LINK_ALERT_REPEAT_S = int(os.environ.get('DOCTOR_LINK_ALERT_REPEAT_S', str(6 * 3600)))  # per camera
PRODUCER_SAMPLE_S = float(os.environ.get('DOCTOR_PRODUCER_SAMPLE_S', '6'))
PRODUCER_MIN_BPS = float(os.environ.get('DOCTOR_PRODUCER_MIN_BPS', '1000'))
PING_COUNT = int(os.environ.get('DOCTOR_PING_COUNT', '10'))

RTSP_STATUS_RE = re.compile(rb'^RTSP/1\.0 (\d{3})')
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

DRY_RUN = False


def cam_ip(cam: dict) -> str:
    return urlparse(cam['rtsp']).hostname or ''


def log(msg: str):
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}', flush=True)


def notify(title: str, message: str):
    if DRY_RUN:
        log(f'[dry-run] would push: {title}: {message}')
        return
    topic = os.environ.get('NTFY_TOPIC')
    if not topic:
        return
    base = os.environ.get('NTFY_URL', 'https://ntfy.sh').rstrip('/')
    try:
        req = urllib.request.Request(f'{base}/{topic}', data=message.encode(), headers={'Title': title})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:
        log(f'notify failed: {type(exc).__name__}')


def http_get(url: str, timeout: float = 10) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def rtsp_status(url: str, timeout: float = 4.0):
    """Status code of an RTSP DESCRIBE (200, 401, 404, ...) or None when nothing answers."""
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 554), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((f'DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\n'
                          'Accept: application/sdp\r\nUser-Agent: camera-doctor\r\n\r\n').encode())
            m = RTSP_STATUS_RE.match(sock.recv(64))
            return int(m.group(1)) if m else None
    except OSError:
        return None


def rtsp_alive(url: str, timeout: float = 4.0) -> bool:
    """Liveness only: any RTSP status line (401 included) counts as alive."""
    return rtsp_status(url, timeout) is not None


def onvif_reboot(host: str, port: int, user: str, password: str) -> str:
    """WS-UsernameToken digest SystemReboot; returns a short result string."""
    nonce = os.urandom(16)
    created = datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
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
    req = urllib.request.Request(f'http://{host}:{port}/onvif/device_service', data=envelope.encode(),
                                 headers={'Content-Type': 'application/soap+xml; charset=utf-8'})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read(500).decode(errors='replace')
            return 'accepted' if ('RebootMessage' in body or resp.status == 200) else f'http {resp.status}'
    except Exception as exc:
        return type(exc).__name__


def parse_ping(text: str):
    """{'sent', 'received', 'loss_pct', 'rtt_avg_ms', 'rtt_max_ms'} from ping's summary, or None if unparseable."""
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


def reboot_allowed(last: dict, now: float, loss_pct: float = LINK_LOSS_PCT,
                   rtt_ms: float = LINK_RTT_MS) -> tuple[bool, str]:
    """May a camera that fails its RTSP probe be sent an ONVIF reboot?

    Yes when nothing fresh is known about its link, or when it answers ping cleanly (the hung-firmware
    signature: IP stack alive, RTSP dead). No when it does not answer ping at all (the reboot command
    could not arrive either) or when the link is lossy/slow (a radio problem a reboot cannot fix).
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


# ---------------------------------------------------------------------------
# Frigate pipeline
# ---------------------------------------------------------------------------
def container_state() -> tuple[str, float]:
    """(docker status, epoch the container last started); status 'unknown' if docker fails."""
    try:
        out = subprocess.run([DOCKER, 'inspect', '-f', '{{.State.Status}} {{.State.StartedAt}}', FRIGATE_CONTAINER],
                             capture_output=True, text=True, timeout=20, check=False)
        status, _, started = out.stdout.strip().partition(' ')
        if out.returncode != 0 or not status:
            return 'unknown', 0.0
        head, _, frac = started.rstrip('Z').partition('.')
        base = datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=UTC).timestamp()
        return status, base + (float('0.' + frac[:6]) if frac else 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 'unknown', 0.0


def iter_log(lines, since: float):
    """(epoch, line) for Frigate/go2rtc log lines newer than `since`.

    Every line from /api/logs starts with a UTC timestamp; a line without one inherits the previous one.
    """
    ts = None
    for line in lines:
        m = TS_RE.match(line)
        if m:
            with suppress(ValueError):
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC).timestamp()
        if ts is not None and ts > since:
            yield ts, line


def parse_frigate_log(lines, since: float) -> dict:
    """Count the failure signatures in Frigate log lines newer than `since`."""
    found = {'dead_thread': [], 'unprocessed': {}, 'estale': 0, 'va_error': 0, 'newest': since}
    for ts, line in iter_log(lines, since):
        found['newest'] = max(found['newest'], ts)
        for key, rx in LOG_SIGNATURES.items():
            m = rx.search(line)
            if not m:
                continue
            if key == 'dead_thread':
                found['dead_thread'].append(m.group(1))
            elif key == 'unprocessed':
                found['unprocessed'][m.group(1)] = found['unprocessed'].get(m.group(1), 0) + 1
            else:
                found[key] += 1
            break
    return found


def parse_go2rtc_log(lines, since: float, ip_to_cam: dict) -> dict:
    """{'timeouts': {camera: n}, 'newest': epoch}: go2rtc producer link errors per camera, matched by the
    camera's IP (':554') in lines newer than `since`."""
    found = {'timeouts': {}, 'newest': since}
    for ts, line in iter_log(lines, since):
        found['newest'] = max(found['newest'], ts)
        if not GO2RTC_LINK_RE.search(line):
            continue
        for ip in GO2RTC_IP_RE.findall(line):
            cam = ip_to_cam.get(ip)
            if cam:
                found['timeouts'][cam] = found['timeouts'].get(cam, 0) + 1
                break
    return found


def fetch_streams():
    """go2rtc's stream table through Frigate's proxy, or None when unavailable."""
    try:
        data = json.loads(http_get(f'{FRIGATE_API}/api/go2rtc/streams'))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def producer_snapshot(streams):
    """stream name -> (producer id, bytes received), or None when the stream has no producer."""
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


def producer_health(snap1, snap2, stream: str, interval_s: float, min_bps: float = PRODUCER_MIN_BPS) -> dict:
    """Is go2rtc's producer for `stream` present, stable and receiving bytes between the two snapshots?"""
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
    """{'degraded': bool, 'measured': bool, 'reasons': [...]} for one camera's link."""
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


def dead_processes(cams_stats: dict, alive: dict, enabled) -> dict:
    """camera -> dead roles among Frigate's own per-camera processes.

    /api/stats reports `capture_pid` (frigate.capture) and `pid` (the tracker). Frigate never restarts
    these when they die (the container's memory cap can OOM-kill a capture process after a post-stall
    frame burst), so a dead one is a pipeline fault whatever the camera link does. `alive` maps
    pid -> bool; unknown pids are not flagged.
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
    script = ('for p in ' + ' '.join(str(p) for p in pids)
              + '; do if [ -d /proc/$p ]; then echo $p 1; else echo $p 0; fi; done')
    try:
        out = subprocess.run([DOCKER, 'exec', FRIGATE_CONTAINER, 'sh', '-c', script],
                             capture_output=True, text=True, timeout=20, check=False)
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
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.stdout.split('\n'):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            res[int(parts[0])] = int(parts[1]) / 1024.0
    return res


def attribute(cam_symptoms: dict, link: dict) -> dict:
    """camera -> 'link' | 'pipeline' | 'unknown' for every camera with a symptom.

    A symptom is the link's fault whenever the link is unhealthy or go2rtc's producer is flapping or
    starved; the pipeline's only when the link was measured healthy and Frigate still gets nothing;
    'unknown' when nothing could be measured (never restarts).
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
    """'ok', 'hung', or the OSError text. Runs in a thread so a dead NAS cannot wedge the run."""
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


def decide(immediate: dict, attribution: dict, strikes: int, wired=WIRED_CAMERAS, solo=()) -> tuple[bool, int]:
    """(restart?, new strike count).

    Only pipeline-attributed symptoms count. A restart needs an immediate fault, or PIPELINE_STRIKES
    consecutive runs of pipeline symptoms that hit >= 2 cameras, a wired camera, or a camera in `solo`
    (its own Frigate process is dead: unambiguous, and Frigate cannot recover it). A single WiFi camera
    can therefore never restart Frigate.
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
        return 'NFS share unhealthy; a restart would just fail again'
    if recent and now - recent[-1] < RESTART_COOLDOWN_S:
        return f'cooldown: last auto-restart {(now - recent[-1]) / 60:.0f} min ago'
    if len(recent) >= RESTART_BUDGET_N:
        return f'budget: already {len(recent)} auto-restarts in 6 h'
    return ''


def link_transition(ls: dict, degraded: bool, now: float, hold_s: float = LINK_HOLD_S) -> tuple[dict, str]:
    """Hysteresis: (new link state, event), event in ('', 'degraded', 'recovered').

    'degraded' fires once the condition has held >= hold_s; 'recovered' once it has been clean >= hold_s
    after a 'degraded'.
    """
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
    """May `key` be pushed again at `now`?"""
    return now - alerts.get(key, 0) >= repeat


def alert(state: dict, key: str, title: str, message: str, now: float,
          repeat: float = ALERT_REPEAT_S, tag: str = '[frigate]') -> bool:
    """Deduped push; True when it was actually sent (or would be, in dry-run)."""
    alerts = state.setdefault('frigate', {}).setdefault('alerts', {})
    if not alert_due(alerts, key, now, repeat):
        return False
    alerts[key] = now
    log(f'{tag} ALERT {title}: {message}')
    notify(title, message)
    return True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
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
    # the old 'budget' alert became the 6-h 'needs_human' one; carry its timestamp over
    fs['alerts'].setdefault('needs_human', fs['alerts'].get('budget', 0))
    return state


# ---------------------------------------------------------------------------
# Link attribution
# ---------------------------------------------------------------------------
def measure_links(state: dict, alive: dict, now: float, verbose: bool) -> dict:
    """camera -> link_health() for every camera: ping, two go2rtc producer samples and the go2rtc read
    timeouts logged since the last run. The measurements are kept in state[cam]['link']['last']."""
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
    parts = [f'{name} ({ip}) link degraded for {(now - since) / 60:.0f} min:']
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


def update_link_state(state: dict, name: str, lh: dict, now: float):
    """Apply the hysteresis to one camera's link, push its degraded/recovered events and judge a dead restream."""
    st = state[name]
    ls, event = link_transition(st['link'], lh['degraded'], now)
    ls.setdefault('alerted', False)
    if event == 'degraded':
        log(f"[link] {name} DEGRADED for >= {LINK_HOLD_S // 60} min: " + '; '.join(lh['reasons']))
        if not st['down_since']:  # the camera layer already pushed DOWN for a dead camera
            st['link'] = ls
            ls['alerted'] = alert(state, f'link_{name}', f'VisionBox: {name} link degraded',
                                  link_message(name, st, now), now, repeat=LINK_ALERT_REPEAT_S, tag='[link]')
    elif event == 'recovered':
        last = ls.get('last', {})
        log(f'[link] {name} link recovered (clean for >= {LINK_HOLD_S // 60} min)')
        if ls.get('alerted'):
            notify(f'VisionBox: {name} link recovered',
                   f"{name} link has been clean for {LINK_HOLD_S // 60} min "
                   f"({last.get('loss_pct') or 0:.0f}% loss, avg RTT "
                   f"{last.get('rtt_avg_ms') or 0:.0f} ms). Do: nothing.")
        ls['alerted'] = False
    st['link'] = ls
    # A restream that has failed FAILS_BEFORE_DOWN runs is only Frigate's fault when the camera's link is clean.
    if st['restream_fails'] == FAILS_BEFORE_DOWN:
        if lh['degraded']:
            log(f"[{name}] camera up but go2rtc restream not serving; link degraded "
                f"({'; '.join(lh['reasons'])}), not blaming Frigate")
        else:
            log(f'[{name}] camera up but go2rtc restream dead; check the frigate container')
            alert(state, f'restream_{name}', 'VisionBox: restream problem',
                  f'{name} camera and its link are fine but its go2rtc restream is not '
                  f'answering. Do: `docker restart frigate` if this repeats.', now, tag=f'[{name}]')


def evaluate_links(state: dict, alive: dict, now: float, verbose: bool) -> dict:
    """Measure every camera's link and raise the link events; camera -> link_health() ({} on failure)."""
    link = {}
    try:
        link = measure_links(state, alive, now, verbose)
        for name in CAMERAS:
            update_link_state(state, name, link[name], now)
    except Exception as exc:  # link measurement must never break the camera checks
        log(f'[link] error: {type(exc).__name__}: {exc}')
    return link


# ---------------------------------------------------------------------------
# Frigate pipeline watchdog
# ---------------------------------------------------------------------------
class Symptoms:
    """What one watchdog run found wrong with Frigate."""

    def __init__(self):
        self.immediate = {}   # key -> text; restarts Frigate at once
        self.general = {}     # key -> text; strike-class symptoms
        self.per_camera = {}  # camera -> [symptom keys], attributed before they count
        self.dead = {}        # camera -> dead Frigate process roles

    def add(self, key: str, text: str, cameras=()):
        self.general[key] = text
        for cam in cameras:
            self.per_camera.setdefault(cam, []).append(key)

    def describe(self, cameras, link: dict) -> str:
        return '; '.join(f"{c}: {'+'.join(self.per_camera[c])}"
                         + (f" [{', '.join(link[c]['reasons'])}]" if link.get(c, {}).get('reasons') else '')
                         for c in cameras)


def container_ready(state: dict, now: float, verbose: bool) -> bool:
    """False while the container is not running or still warming up; both reset the strikes."""
    fs = state['frigate']
    status, started = container_state()
    if status != 'running':
        fs['down_runs'] += 1
        fs['strikes'] = 0
        if fs['down_runs'] >= 2:
            alert(state, 'not_running', 'VisionBox: Frigate container not running',
                  f"docker reports the frigate container is '{status}'. Docker's restart "
                  f'policy should bring it back. Do: run `docker ps -a` and '
                  f'`docker start frigate` if it stays down.', now)
        return False
    fs['down_runs'] = 0
    if now - started < WARMUP_S:
        if verbose:
            log(f'[frigate] container is {(now - started) / 60:.1f} min old; skipping pipeline checks')
        fs.update(strikes=0, frame_hash={})
        return False
    return True


def check_nfs(state: dict, now: float) -> list:
    """Problems with the NFS export. Nothing Frigate does can work on a missing or stale one, and
    restarting it would only reproduce the failure."""
    problems = []
    if not os.path.ismount(NFS_MOUNT):
        problems.append(f'{NFS_MOUNT} is not mounted')
    else:
        for sub in NFS_PROBE_DIRS:
            res = nfs_write_probe(NFS_MOUNT / sub)
            if res != 'ok':
                problems.append(f'{sub}: {res}')
    if problems:
        alert(state, 'nfs', 'VisionBox: NFS share problem',
              '; '.join(problems) + '. Frigate cannot record until the NAS export is '
              'healthy; not restarting it. Do: check the NAS / remount the export.', now)
    return problems


def fetch_stats():
    """(Frigate's per-camera stats, '') or (None, error name) when the API failed twice."""
    err = ''
    for attempt in range(2):
        try:
            return json.loads(http_get(f'{FRIGATE_API}/api/stats', timeout=10 + 5 * attempt)).get('cameras', {}), ''
        except Exception as exc:
            err = type(exc).__name__
            if attempt == 0:
                time.sleep(5)
    return None, err


def scan_frigate_log(state: dict, now: float, sym: Symptoms):
    """Immediate faults and strike symptoms from Frigate's own log since the last run."""
    fs = state['frigate']
    cursor = fs.get('log_cursor')
    since = now - LOG_WINDOW_S if cursor is None else max(cursor, now - 3600)
    try:
        payload = json.loads(http_get(f'{FRIGATE_API}/api/logs/frigate?start=-1500', timeout=15))
        found = parse_frigate_log(payload.get('lines', []), since)
    except Exception as exc:
        log(f'[frigate] /api/logs unreachable ({type(exc).__name__}); log signatures skipped this run')
        return
    fs['log_cursor'] = found['newest']
    if found['dead_thread']:
        sym.immediate['dead_thread'] = 'dead thread: ' + ', '.join(sorted(set(found['dead_thread'])))
    if found['va_error']:
        sym.immediate['va_error'] = (f"{found['va_error']} VAAPI device errors "
                                     f'(GPU render node missing from the container?)')
    unproc = sum(found['unprocessed'].values())
    if unproc >= UNPROCESSED_SUSTAINED:
        sym.add('unprocessed', f"{unproc} 'unprocessed recording segments' warnings "
                               f"({', '.join(sorted(found['unprocessed']))})", found['unprocessed'])
    if found['estale']:
        alert(state, 'estale', 'VisionBox: NFS stale file handles',
              f"Frigate logged {found['estale']} 'Stale file handle' errors: the NAS "
              f'export changed underneath it. Watching for a dead thread. Do: nothing yet; '
              f'an auto-restart follows if a thread dies.', now)


def camera_symptoms(state: dict, cams: dict, restream_up: dict, now: float, sym: Symptoms):
    """Per-camera pipeline symptoms from /api/stats, the container's process table and latest.jpg."""
    fs = state['frigate']
    try:
        cfg_cams = json.loads(http_get(f'{FRIGATE_API}/api/config')).get('cameras', {})
    except Exception:
        cfg_cams = {}
    enabled = [c for c in CAMERAS if cfg_cams.get(c, {}).get('enabled', True)]
    stats = {c: cams.get(c) or {} for c in enabled}
    fps = {c: stats[c].get('camera_fps') or 0 for c in enabled}

    # Frigate's own per-camera processes: a dead one is a pipeline fault, Frigate never restarts them.
    pids = [p for c in enabled for p in (stats[c].get('capture_pid'), stats[c].get('pid')) if p]
    sym.dead = dead_processes(cams, probe_pids(pids), enabled)
    if sym.dead:
        sym.add('process_dead', 'Frigate camera process dead (Frigate does not restart it): '
                + ', '.join(f"{c} ({'+'.join(r)})" for c, r in sym.dead.items()), sym.dead)

    # early warning: frigate.capture ballooning towards the container's memory cap
    rss = probe_rss([stats[c].get('capture_pid') for c in enabled])
    for c in enabled:
        cp = stats[c].get('capture_pid')
        mb = rss.get(int(cp)) if cp else None
        if mb and mb >= CAPTURE_RSS_ALERT_MB:
            alert(state, f'rss_{c}', f'VisionBox: {c} capture process ballooning',
                  f'frigate.capture for {c} uses {mb:.0f} MB (container cap 4 GB). After a WiFi '
                  f'stall this process can balloon and be OOM-killed, which silently stops the '
                  f'camera. Do: nothing now; the watchdog restarts Frigate if the process dies.',
                  now, repeat=HUMAN_REPEAT_S)

    zero = [c for c in enabled if restream_up.get(c) and fps[c] == 0]
    if zero:
        sym.add('zero_fps', 'restream up but Frigate reads 0 fps: ' + ', '.join(zero), zero)
    # camera answers directly but go2rtc has no usable restream for it (404 / no answer)
    no_restream = [c for c in enabled if restream_up.get(c) is False]
    if no_restream:
        sym.add('no_restream', 'camera up but go2rtc restream not serving: ' + ', '.join(no_restream), no_restream)

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
        sym.add('frozen', 'latest.jpg unchanged for 5 min on a camera reporting frames: ' + ', '.join(frozen), frozen)


def auto_restart(state: dict, why: str, now: float):
    fs = state['frigate']
    log(f'[frigate] AUTO-RESTART: {why}')
    if DRY_RUN:
        log('[frigate] dry-run: not restarting')
        return
    # Persist the restart before issuing it so a killed run cannot cause a loop.
    fs['restarts'].append(now)
    fs.update(strikes=0, frame_hash={}, log_cursor=now, verify_after_restart=True)
    STATE_FILE.write_text(json.dumps(state))
    try:
        subprocess.run([DOCKER, 'restart', FRIGATE_CONTAINER], capture_output=True, text=True, timeout=180, check=True)
        result = 'done'
    except (OSError, subprocess.SubprocessError) as exc:
        result = f'FAILED ({type(exc).__name__})'
        fs['verify_after_restart'] = False
    log(f'[frigate] docker restart {FRIGATE_CONTAINER}: {result}')
    notify('VisionBox: Frigate auto-restarted',
           f'{why}. docker restart: {result}. Health will be re-checked in ~10 min. '
           f"Do: nothing unless a 'keeps failing' push follows.")


def frigate_watchdog(state: dict, restream_up: dict, now: float, verbose: bool, link: dict | None = None):
    """One watchdog run: collect Frigate's symptoms, attribute them, then strike, alert or restart."""
    fs = state.setdefault('frigate', {})
    for key, default in FRIGATE_STATE_DEFAULTS.items():
        fs.setdefault(key, type(default)() if isinstance(default, (dict, list)) else default)
    link = link or {}
    if not container_ready(state, now, verbose):
        return
    nfs_problems = check_nfs(state, now)

    sym = Symptoms()
    cams, err = fetch_stats()
    if cams is None:
        # A dead API is a pipeline symptom whatever the links do, but a strike rather than an immediate restart.
        sym.add('api_down', f'/api/stats unreachable twice ({err}) while the container runs', WIRED_CAMERAS)
    scan_frigate_log(state, now, sym)
    if cams is not None:
        camera_symptoms(state, cams, restream_up, now, sym)

    attribution = attribute(sym.per_camera, link)
    if 'api_down' in sym.general:
        for c in WIRED_CAMERAS:
            attribution[c] = 'pipeline'
    for c in sym.dead:  # a dead Frigate process is never the link's fault
        attribution[c] = 'pipeline'
    fs['attribution'] = attribution
    restart, fs['strikes'] = decide(sym.immediate, attribution, fs['strikes'], solo=tuple(sym.dead))
    pipeline_cams = sorted(c for c, a in attribution.items() if a == 'pipeline')
    link_cams = sorted(c for c, a in attribution.items() if a != 'pipeline')
    if verbose:
        log(f"[frigate] immediate={sym.immediate or '-'} symptoms={sym.general or '-'} "
            f"attribution={attribution or '-'} strikes={fs['strikes']} restart={restart}")

    if not restart:
        if link_cams:
            log(f'[frigate] link-attributed, no restart: {sym.describe(link_cams, link)}')
        if pipeline_cams:
            log(f"[frigate] strike {fs['strikes']} (pipeline): {sym.describe(pipeline_cams, link)}")
            if fs['strikes'] >= HUMAN_AFTER_STRIKES:
                alert(state, 'needs_human', 'VisionBox: Frigate keeps failing',
                      f"Pipeline fault for {fs['strikes'] * 5} min on {', '.join(pipeline_cams)} "
                      f'({sym.describe(pipeline_cams, link)}); a single WiFi camera does not trigger an '
                      f"auto-restart. Do: open Frigate's System/Logs page; run "
                      f'`docker restart frigate` yourself if its recordings are missing.',
                      now, repeat=HUMAN_REPEAT_S)
        elif fs.pop('verify_after_restart', False):
            note = (f" Link still degraded on {', '.join(link_cams)} (camera side; Frigate "
                    f'reconnects on its own).' if link_cams else '')
            log('[frigate] healthy after auto-restart')
            notify('VisionBox: Frigate healthy again',
                   f'Pipeline checks are clean after the auto-restart.{note} Do: nothing.')
        return

    reasons = list(sym.immediate.values())
    if pipeline_cams:
        reasons.append(sym.describe(pipeline_cams, link))
    why = '; '.join(reasons)
    blocked = restart_blocked(fs, now, nfs_problems)
    if blocked:
        log(f'[frigate] restart wanted ({why}) but not doing it: {blocked}')
        alert(state, 'needs_human', 'VisionBox: Frigate keeps failing',
              f"{why}. Auto-restart blocked: {blocked}; needs a human. Do: check Frigate's "
              f'System/Logs page and the NAS, then `docker restart frigate` yourself.',
              now, repeat=HUMAN_REPEAT_S)
        return
    auto_restart(state, why, now)


# ---------------------------------------------------------------------------
# Camera layer and the run itself
# ---------------------------------------------------------------------------
def probe_cameras(state: dict, now: float, verbose: bool) -> tuple[dict, dict, list]:
    """Probe every camera and its restream; (camera -> alive, camera -> restream up, cameras wanting a reboot).

    A camera that is down has no restream entry: that would be the same outage seen through go2rtc.
    """
    alive_map, restream_up, reboot_wanted = {}, {}, []
    for name, cam in CAMERAS.items():
        st = state[name]
        alive = rtsp_alive(cam['rtsp'])
        alive_map[name] = alive
        if verbose:
            log(f"[{name}] camera={'up' if alive else 'DOWN'}")
        if alive:
            if st['down_since']:
                mins = (now - st['down_since']) / 60
                log(f'[{name}] RECOVERED after {mins:.0f} min')
                if st['notified']:
                    notify(f'VisionBox: {name} recovered',
                           f'{name} is answering again after {mins:.0f} min down. Do: nothing.')
            st.update(fails=0, down_since=None, notified=False)
            r_status = rtsp_status(f"rtsp://127.0.0.1:8554/{cam['restream']}")
            r_ok = r_status in RESTREAM_OK_STATUS
            restream_up[name] = r_ok
            if verbose:
                log(f"[{name}] restream={'up' if r_ok else 'DOWN'} (status {r_status})")
            st['restream_fails'] = 0 if r_ok else st['restream_fails'] + 1
        else:
            st['fails'] += 1
            if st['fails'] == FAILS_BEFORE_DOWN:
                st['down_since'] = now - FAILS_BEFORE_DOWN * 300
                log(f'[{name}] DOWN ({FAILS_BEFORE_DOWN} consecutive probe failures)')
                notify(f'VisionBox: {name} DOWN',
                       f'{name} RTSP has not answered for ~15 min. '
                       f'Attempting ONVIF reboot. Do: if it stays down, power-cycle the camera.')
                st['notified'] = True
            if st['fails'] >= FAILS_BEFORE_DOWN and now - st['last_reboot'] > REBOOT_COOLDOWN_S:
                reboot_wanted.append(name)
    return alive_map, restream_up, reboot_wanted


def reboot_cameras(state: dict, names: list, now: float):
    """ONVIF-reboot the DOWN cameras whose link measurement looks like hung firmware rather than a dead radio."""
    for name in names:
        st = state[name]
        ok, why = reboot_allowed(st['link'].get('last') or {}, now)
        if not ok:
            if now - st.get('reboot_skipped_at', 0) > REBOOT_COOLDOWN_S:
                st['reboot_skipped_at'] = now
                log(f'[{name}] not sending ONVIF reboot: {why}')
            continue
        if DRY_RUN:
            log(f'[{name}] dry-run: would send ONVIF reboot')
            continue
        st['last_reboot'] = now
        result = onvif_reboot(*CAMERAS[name]['onvif'])
        log(f'[{name}] ONVIF reboot attempt: {result}')


def trim_log():
    """Keep the cron-appended log bounded."""
    try:
        if LOG_FILE.stat().st_size > 1_000_000:
            lines = LOG_FILE.read_text().splitlines()[-500:]
            LOG_FILE.write_text('\n'.join(lines) + '\n')
    except OSError:
        pass


def run(verbose: bool):
    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        state = {}
    state = normalise_state(state)
    now = time.time()

    alive, restream_up, reboot_wanted = probe_cameras(state, now, verbose)
    link = evaluate_links(state, alive, now, verbose)
    reboot_cameras(state, reboot_wanted, now)  # after the link measurement: it decides whether a reboot can help
    try:
        frigate_watchdog(state, restream_up, now, verbose, link)
    except Exception as exc:  # the camera checks above must never be lost to a watchdog bug
        log(f'[frigate] watchdog error: {type(exc).__name__}: {exc}')

    STATE_FILE.write_text(json.dumps(state))
    trim_log()


def main():
    parser = argparse.ArgumentParser(description='Camera and Frigate watchdog; see the module docstring.')
    parser.add_argument('--verbose', action='store_true', help='print every probe result')
    parser.add_argument('--dry-run', action='store_true',
                        help='decide but never restart Frigate, reboot a camera or send ntfy pushes')
    args = parser.parse_args()
    global DRY_RUN
    DRY_RUN = args.dry_run

    # Hard ceiling so a wedged probe pile-up cannot outlive the cron interval (the lock prevents overlap).
    signal.alarm(280)
    STATE_FILE.parent.mkdir(exist_ok=True)
    with STATE_FILE.with_suffix('.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # previous run still going
        run(args.verbose)


if __name__ == '__main__':
    main()
