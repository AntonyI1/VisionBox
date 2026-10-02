#!/usr/bin/env python3
"""Tests for scripts/camera_doctor.py.

Policy helpers are tested directly. frigate_watchdog() and the camera layer are replayed end-to-end with the
network and docker calls stubbed, using measurements recorded on the box during the 2026-09-13 WiFi outage.
Run from the repo root with `venv/bin/python3 -m unittest tests.test_camera_doctor` or pytest.
"""
import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TMP = tempfile.mkdtemp(prefix='camera_doctor_test_')
os.environ.setdefault('DOCTOR_STATE_FILE', str(Path(_TMP) / 'test_state.json'))
os.environ.setdefault('DOCTOR_ENV_FILE', '/nonexistent/.env')  # the tests must not need credentials
spec = importlib.util.spec_from_file_location('camera_doctor', HERE.parent / 'scripts' / 'camera_doctor.py')
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

NOW = 1789325703.0  # 2026-09-13 18:55:03 UTC

# The state file as it was on the box at 18:50 UTC that day, written before link attribution existed.
REAL_STATE = {
    'front_door': {'fails': 0, 'down_since': None, 'last_reboot': 1788838801.350269, 'notified': False,
                   'restream_fails': 0},
    'front_garage': {'fails': 0, 'down_since': None, 'last_reboot': 1789264801.6268091, 'notified': False,
                     'restream_fails': 0},
    'backyard': {'fails': 0, 'down_since': None, 'last_reboot': 1788838801.350269, 'notified': False,
                 'restream_fails': 0},
    'backyard_gate': {'fails': 0, 'down_since': None, 'last_reboot': 1789110601.8853848, 'notified': False,
                      'restream_fails': 0},
    'frigate': {
        'strikes': 0,
        'restarts': [1789314601.3453732, 1789316402.0242512, 1789319701.2236552],
        'frame_hash': {'front_door': '26e7ec236b5a094e192e93e3e9fc9421',
                       'front_garage': 'd5638c15191f5fb1cbf728c62ca5921f',
                       'backyard': 'c7b3c8f612faa361ff1a6441b9c4f1fc',
                       'backyard_gate': 'e87e67724a7ae608868cf6b5501747f6'},
        'alerts': {'estale': 1789163701.618383, 'budget': 1789323601.3912013},
        'down_runs': 0,
        'log_cursor': 1789325351.0,
    },
}

# `ping -c 5 -i 0.2 -W 1` output recorded at 18:5x UTC that day.
PING_FRONT_GARAGE = ('5 packets transmitted, 1 received, 80% packet loss, time 821ms\n'
                     'rtt min/avg/max/mdev = 184.946/184.946/184.946/0.000 ms\n')
PING_BACKYARD = ('5 packets transmitted, 5 received, 0% packet loss, time 814ms\n'
                 'rtt min/avg/max/mdev = 14.645/188.890/504.091/187.323 ms, pipe 3\n')
PING_BACKYARD_GATE = ('5 packets transmitted, 2 received, 60% packet loss, time 813ms\n'
                      'rtt min/avg/max/mdev = 42.960/113.941/184.922/70.981 ms\n')
PING_FRONT_DOOR = ('5 packets transmitted, 5 received, 0% packet loss, time 815ms\n'
                   'rtt min/avg/max/mdev = 0.615/0.814/1.158/0.192 ms\n')
PING_ALL_LOST = '5 packets transmitted, 0 received, 100% packet loss, time 4080ms\n'
GO2RTC_LINE = ('2026-09-13 19:00:22.808735946  12:00:22.808 WRN '
               'github.com/AlexxIT/go2rtc/internal/streams/producer.go:170 '
               '> error="read tcp 172.18.0.2:58974->192.168.1.249:554: i/o timeout" url=rtsp://u:p@192.168.1.249:554/stream2')

HEALTHY = {'sampled': True, 'present': True, 'reconnected': False, 'bytes': 300000, 'bps': 50000.0,
           'starved': False, 'id': 7}
CLEAN_PING = cd.parse_ping(PING_FRONT_DOOR)
OK_LINK = cd.link_health(True, CLEAN_PING, HEALTHY, 0)
ALL_UP = dict.fromkeys(cd.CAMERAS, True)
LINK_TODAY = {'front_door': OK_LINK,
              'front_garage': cd.link_health(True, cd.parse_ping(PING_FRONT_GARAGE), HEALTHY, 12),
              'backyard': cd.link_health(True, cd.parse_ping(PING_BACKYARD), HEALTHY, 1),
              'backyard_gate': cd.link_health(True, cd.parse_ping(PING_BACKYARD_GATE), HEALTHY, 0)}
LINK_CLEAN = dict.fromkeys(cd.CAMERAS, OK_LINK)
PID_STATS = {'front_garage': {'capture_pid': 1187, 'pid': 1084}, 'backyard': {'capture_pid': 894, 'pid': 837}}


def real_state():
    return cd.normalise_state(copy.deepcopy(REAL_STATE))


class ParsePing(unittest.TestCase):
    def test_real_outputs(self):
        p = cd.parse_ping(PING_FRONT_GARAGE)
        self.assertEqual((p['sent'], p['received']), (5, 1))
        self.assertAlmostEqual(p['loss_pct'], 80.0)
        self.assertAlmostEqual(p['rtt_avg_ms'], 184.946)
        p = cd.parse_ping(PING_BACKYARD)
        self.assertAlmostEqual(p['loss_pct'], 0.0)
        self.assertAlmostEqual(p['rtt_avg_ms'], 188.89)
        self.assertAlmostEqual(p['rtt_max_ms'], 504.091)
        p = cd.parse_ping(PING_ALL_LOST)
        self.assertEqual(p['loss_pct'], 100.0)
        self.assertIsNone(p['rtt_avg_ms'])
        self.assertIsNone(cd.parse_ping('ping: connect: Network is unreachable'))
        self.assertIsNone(cd.parse_ping(''))


class ProducerHealth(unittest.TestCase):
    def test_increasing_is_healthy(self):
        s1 = {'front_garage': (2757, 1088648)}
        s2 = {'front_garage': (2757, 1384836)}  # real sample: +296188 B in 6 s
        h = cd.producer_health(s1, s2, 'front_garage', 6.0)
        self.assertTrue(h['sampled'] and h['present'] and not h['starved'] and not h['reconnected'])
        self.assertAlmostEqual(h['bps'], 296188 / 6.0)

    def test_not_increasing_is_starved(self):
        h = cd.producer_health({'x': (1, 500)}, {'x': (1, 500)}, 'x', 6.0)
        self.assertTrue(h['starved'])
        h = cd.producer_health({'x': (1, 500)}, {'x': (1, 500 + 5000)}, 'x', 6.0)  # 833 B/s < 1000
        self.assertTrue(h['starved'])

    def test_reconnect_and_missing(self):
        h = cd.producer_health({'x': (1, 5000)}, {'x': (2, 100)}, 'x', 6.0)
        self.assertTrue(h['reconnected'] and h['starved'])
        h = cd.producer_health({'x': None}, {'x': None}, 'x', 6.0)
        self.assertTrue(h['sampled'] and not h['present'] and h['starved'])
        h = cd.producer_health({'x': (1, 5)}, {'x': None}, 'x', 6.0)
        self.assertTrue(h['reconnected'])

    def test_unsampled(self):
        h = cd.producer_health(None, None, 'x', 6.0)
        self.assertFalse(h['sampled'])
        self.assertFalse(h['starved'])

    def test_snapshot_uses_bytes_recv_or_recv(self):
        snap = cd.producer_snapshot({'a': {'producers': [{'id': 4, 'bytes_recv': 10}]},
                                     'b': {'producers': [{'id': 5, 'recv': 20}]},
                                     'c': {'producers': []}, 'd': None})
        self.assertEqual(snap, {'a': (4, 10), 'b': (5, 20), 'c': None, 'd': None})
        self.assertIsNone(cd.producer_snapshot(None))


class LinkHealthAndAttribution(unittest.TestCase):
    def test_lossy_link_with_timeouts_is_degraded(self):
        lh = cd.link_health(True, cd.parse_ping(PING_FRONT_GARAGE), HEALTHY, 12)
        self.assertTrue(lh['degraded'])
        self.assertIn('ping loss 80%', lh['reasons'])
        self.assertTrue(any('go2rtc read timeouts' in r for r in lh['reasons']))

    def test_high_rtt_alone_is_degraded(self):
        lh = cd.link_health(True, cd.parse_ping(PING_BACKYARD), HEALTHY, 0)
        self.assertTrue(lh['degraded'])
        self.assertEqual(lh['reasons'], ['ping avg RTT 189 ms'])

    def test_clean_link_with_starved_producer_is_degraded(self):
        starved = dict(HEALTHY, bps=0.0, bytes=0, starved=True)
        lh = cd.link_health(True, CLEAN_PING, starved, 0)
        self.assertTrue(lh['degraded'])

    def test_clean_link_healthy_producer_is_not_degraded(self):
        self.assertFalse(OK_LINK['degraded'])
        self.assertTrue(OK_LINK['measured'])

    def test_attribute(self):
        link = {'front_garage': cd.link_health(True, cd.parse_ping(PING_FRONT_GARAGE), HEALTHY, 3),
                'front_door': OK_LINK,
                'backyard': {'degraded': False, 'measured': False, 'reasons': []}}
        att = cd.attribute({'front_garage': ['zero_fps'], 'front_door': ['zero_fps'],
                            'backyard': ['frozen'], 'backyard_gate': ['frozen']}, link)
        self.assertEqual(att, {'front_garage': 'link', 'front_door': 'pipeline',
                               'backyard': 'unknown', 'backyard_gate': 'unknown'})
        self.assertEqual(cd.attribute({}, link), {})


class Decide(unittest.TestCase):
    def test_single_link_cam_never_restarts(self):
        # 17:10 "strike 1" then 17:15 "AUTO-RESTART: restream up but Frigate reads 0 fps: front_garage"
        r, s = cd.decide({}, {'front_garage': 'link'}, 0)
        self.assertEqual((r, s), (False, 0))
        r, s = cd.decide({}, {'front_garage': 'link'}, s)
        self.assertEqual((r, s), (False, 0))

    def test_two_different_link_cams_on_consecutive_runs_never_restart(self):
        # 03:05 frozen backyard_gate, 03:10 frozen front_garage: a global strike counter restarted on that
        r, s = cd.decide({}, {'backyard_gate': 'link'}, 0)
        r2, s2 = cd.decide({}, {'front_garage': 'link'}, s)
        self.assertFalse(r or r2)
        self.assertEqual(s2, 0)

    def test_two_link_cams_at_once_never_restart(self):
        r, s = cd.decide({}, {'front_garage': 'link', 'backyard': 'link'}, 1)
        self.assertEqual((r, s), (False, 0))

    def test_pipeline_wide_zero_fps_restarts_on_second_run(self):
        att = dict.fromkeys(('front_door', 'front_garage', 'backyard', 'backyard_gate'), 'pipeline')
        r, s = cd.decide({}, att, 0)
        self.assertEqual((r, s), (False, 1))
        r, s = cd.decide({}, att, s)
        self.assertEqual((r, s), (True, 2))

    def test_two_pipeline_cams_restart(self):
        self.assertTrue(cd.decide({}, {'backyard': 'pipeline', 'backyard_gate': 'pipeline'}, 1)[0])

    def test_wired_pipeline_alone_restarts(self):
        self.assertTrue(cd.decide({}, {'front_door': 'pipeline'}, 1)[0])

    def test_single_wifi_pipeline_cam_never_restarts(self):
        s = 0
        for _ in range(6):
            r, s = cd.decide({}, {'backyard': 'pipeline'}, s)
            self.assertFalse(r)
        self.assertEqual(s, 6)  # strikes keep climbing: "needs a human" after HUMAN_AFTER_STRIKES

    def test_mixed_link_and_single_pipeline_no_restart(self):
        r, s = cd.decide({}, {'backyard': 'link', 'front_garage': 'pipeline'}, 1)
        self.assertEqual((r, s), (False, 2))

    def test_unknown_never_counts(self):
        r, s = cd.decide({}, {'backyard': 'unknown', 'front_door': 'unknown'}, 5)
        self.assertEqual((r, s), (False, 0))

    def test_immediate_faults_restart_now(self):
        self.assertTrue(cd.decide({'dead_thread': 'dead thread: event_cleanup'}, {}, 0)[0])
        self.assertTrue(cd.decide({'va_error': '3 VAAPI device errors'}, {}, 0)[0])
        self.assertTrue(cd.decide({'api_down': '/api/stats unreachable twice'}, {}, 0)[0])
        # a dead thread restarts even when every camera symptom is link-attributed
        self.assertTrue(cd.decide({'dead_thread': 'x'}, {'front_garage': 'link'}, 0)[0])


class RestartBlocked(unittest.TestCase):
    def test_cooldown_budget_nfs_and_ageing(self):
        fs = {'restarts': [NOW - 600]}
        self.assertTrue(cd.restart_blocked(fs, NOW, []).startswith('cooldown'))
        fs = {'restarts': [NOW - 5 * 3600, NOW - 4 * 3600, NOW - 3 * 3600]}
        self.assertTrue(cd.restart_blocked(fs, NOW, []).startswith('budget'))
        fs = {'restarts': [NOW - 7 * 3600, NOW - 4 * 3600, NOW - 3 * 3600]}
        self.assertEqual(cd.restart_blocked(fs, NOW, []), '')
        self.assertEqual(len(fs['restarts']), 2)  # the 7 h old one aged out
        self.assertIn('NFS', cd.restart_blocked({'restarts': []}, NOW, ['clips: hung']))

    def test_real_state_budget_exhausted(self):
        fs = copy.deepcopy(REAL_STATE['frigate'])
        # at 18:55 UTC the restarts at 15:50, 16:20 and 17:15 are all inside the 6 h window
        self.assertTrue(cd.restart_blocked(fs, NOW, []).startswith('budget'))


class LinkHysteresis(unittest.TestCase):
    def run_seq(self, ls, seq, t0):
        events = []
        for i, d in enumerate(seq):
            ls, ev = cd.link_transition(ls, d, t0 + 300 * i)
            events.append(ev)
        return ls, events

    def test_degraded_push_after_15_min_then_recovery_after_15_min_clean(self):
        ls = dict(cd.LINK_STATE_DEFAULTS)
        # bad at t0, +5, +10 -> nothing; +15 min -> 'degraded'; still bad -> nothing
        ls, ev = self.run_seq(ls, [True, True, True, True, True], NOW)
        self.assertEqual(ev, ['', '', '', 'degraded', ''])
        self.assertEqual(ls['status'], 'degraded')
        # clean at +25, +30, +35 -> nothing; +40 (15 min clean) -> 'recovered'
        ls, ev = self.run_seq(ls, [False, False, False, False, False], NOW + 1500)
        self.assertEqual(ev, ['', '', '', 'recovered', ''])
        self.assertEqual(ls['status'], 'ok')

    def test_short_blips_never_alert(self):
        ls = dict(cd.LINK_STATE_DEFAULTS)
        ls, ev = self.run_seq(ls, [True, True, False, True, True, False, True], NOW)
        self.assertEqual(set(ev), {''})
        self.assertEqual(ls['status'], 'ok')

    def test_cron_jitter_tolerated(self):
        ls = dict(cd.LINK_STATE_DEFAULTS)
        ls, _ = cd.link_transition(ls, True, NOW)
        ls, ev = cd.link_transition(ls, True, NOW + 900 - 12)  # run started 12 s early
        self.assertEqual(ev, 'degraded')

    def test_recovery_only_after_degraded(self):
        ls = dict(cd.LINK_STATE_DEFAULTS)
        ls, ev = self.run_seq(ls, [False] * 6, NOW)
        self.assertEqual(set(ev), {''})


class AlertDedupe(unittest.TestCase):
    def test_windows(self):
        alerts = {'budget': NOW - 1800, 'needs_human': NOW - 1800, 'link_front_garage': NOW - 3599}
        self.assertFalse(cd.alert_due(alerts, 'budget', NOW))  # 1 h default
        self.assertTrue(cd.alert_due(alerts, 'budget', NOW + 1801))
        self.assertFalse(cd.alert_due(alerts, 'needs_human', NOW, cd.HUMAN_REPEAT_S))
        self.assertFalse(cd.alert_due(alerts, 'needs_human', NOW + 5 * 3600, cd.HUMAN_REPEAT_S))
        self.assertTrue(cd.alert_due(alerts, 'needs_human', NOW + 6 * 3600 - 1799, cd.HUMAN_REPEAT_S))
        self.assertFalse(cd.alert_due(alerts, 'link_front_garage', NOW))
        self.assertTrue(cd.alert_due(alerts, 'link_front_garage', NOW + 1))
        self.assertTrue(cd.alert_due({}, 'anything', NOW))

    def test_alert_helper_records_and_dedupes(self):
        sent = []
        cd.notify = lambda t, m: sent.append(t)
        try:
            state = {'frigate': {'alerts': {}}}
            self.assertTrue(cd.alert(state, 'k', 'T', 'm', NOW))
            self.assertFalse(cd.alert(state, 'k', 'T', 'm', NOW + 100))
            self.assertTrue(cd.alert(state, 'k', 'T', 'm', NOW + 3600))
            self.assertEqual(sent, ['T', 'T'])
        finally:
            cd.notify = ORIG_NOTIFY

    def test_human_alert_seeded_from_old_budget_key(self):
        st = real_state()
        self.assertEqual(st['frigate']['alerts']['needs_human'], REAL_STATE['frigate']['alerts']['budget'])
        self.assertFalse(cd.alert_due(st['frigate']['alerts'], 'needs_human', NOW, cd.HUMAN_REPEAT_S))


class StateCompat(unittest.TestCase):
    def test_real_state_loads_and_keeps_every_old_key(self):
        st = real_state()

        def old_keys_preserved(old, new, path):
            for k, v in old.items():
                self.assertIn(k, new, f'{path}.{k} dropped')
                if isinstance(v, dict):
                    old_keys_preserved(v, new[k], f'{path}.{k}')
                else:
                    self.assertEqual(new[k], v, f'{path}.{k} changed')

        old_keys_preserved(REAL_STATE, st, 'state')
        for cam in cd.CAMERAS:
            self.assertEqual(st[cam]['link']['status'], 'ok')
            self.assertIsNone(st[cam]['link']['bad_since'])
        self.assertEqual(st['frigate']['attribution'], {})
        self.assertIn('needs_human', st['frigate']['alerts'])
        json.dumps(st)  # still serialisable

    def test_empty_and_garbage_state(self):
        st = cd.normalise_state({})
        self.assertEqual(st['front_door']['fails'], 0)
        self.assertEqual(st['frigate']['restarts'], [])
        st = cd.normalise_state(None)
        self.assertIn('frigate', st)

    def test_defaults_are_not_shared_between_cameras(self):
        st = cd.normalise_state({})
        st['front_door']['link']['last']['x'] = 1
        self.assertNotIn('x', st['backyard']['link']['last'])


class EnvFile(unittest.TestCase):
    def test_stdlib_parser_never_overrides_the_environment(self):
        path = Path(_TMP) / 'parse.env'
        path.write_text('# comment\nexport DOCTOR_T_A="quoted"\nDOCTOR_T_B = plain \nDOCTOR_T_C=file\nnot a pair\n')
        os.environ['DOCTOR_T_C'] = 'preset'
        orig = cd.load_dotenv
        cd.load_dotenv = None
        try:
            cd._load_env_file(path)
            cd._load_env_file(Path('/nonexistent/.env'))  # unreadable: silently ignored
            self.assertEqual([os.environ.get(k) for k in ('DOCTOR_T_A', 'DOCTOR_T_B', 'DOCTOR_T_C')],
                             ['quoted', 'plain', 'preset'])
        finally:
            cd.load_dotenv = orig
            for k in ('DOCTOR_T_A', 'DOCTOR_T_B', 'DOCTOR_T_C'):
                os.environ.pop(k, None)


class LogParsers(unittest.TestCase):
    def test_iter_log_skips_old_lines_and_inherits_timestamps(self):
        since = cd.datetime(2026, 9, 11, 21, 52, tzinfo=cd.UTC).timestamp()
        lines = ['2026-09-11 21:50:00.0  too old', '2026-09-11 21:54:50.1  new', 'no timestamp: inherits',
                 '2026-99-99 00:00:00  unparseable stamp: inherits too']
        kept = list(cd.iter_log(lines, since))
        self.assertEqual([line for _, line in kept], lines[1:])
        self.assertEqual(len({ts for ts, _ in kept}), 1)
        self.assertEqual(list(cd.iter_log(['no timestamp at all'], since)), [])

    def test_parse_frigate_log(self):
        lines = ['2026-09-11 21:54:50.1  [2026-09-11 14:54:50] frigate.record.maintainer WARNING : Too many '
                 'unprocessed recording segments in cache for backyard. This likely indicates an issue',
                 '2026-09-11 21:54:55.1  Exception in thread event_cleanup:',
                 '2026-09-11 21:54:56.1  OSError: [Errno 116] Stale file handle',
                 'no timestamp line inherits previous',
                 '2026-09-11 21:50:00.0  Exception in thread too_old:']
        since = cd.datetime(2026, 9, 11, 21, 52, tzinfo=cd.UTC).timestamp()
        f = cd.parse_frigate_log(lines, since)
        self.assertEqual(f['dead_thread'], ['event_cleanup'])
        self.assertEqual(f['unprocessed'], {'backyard': 1})
        self.assertEqual(f['estale'], 1)
        self.assertGreater(f['newest'], since)

    def test_parse_go2rtc_log(self):
        ipmap = {'192.168.1.249': 'front_garage', '192.168.1.101': 'backyard'}
        since = cd.datetime(2026, 9, 13, 18, 0, tzinfo=cd.UTC).timestamp()
        backyard_line = GO2RTC_LINE.replace('249', '101').replace('i/o timeout', 'RTP header size insufficient')
        g = cd.parse_go2rtc_log([GO2RTC_LINE, backyard_line,
                                 '2026-09-13 17:15:11.7  10:15:11.748 INF go2rtc platform=linux/amd64'], since, ipmap)
        self.assertEqual(g['timeouts'], {'front_garage': 1, 'backyard': 1})
        g = cd.parse_go2rtc_log([GO2RTC_LINE], since + 10 * 3600, ipmap)  # older than the cursor
        self.assertEqual(g['timeouts'], {})


class DryRunIsInert(unittest.TestCase):
    def test_notify_does_not_open_a_connection_in_dry_run(self):
        calls = []
        orig = cd.urllib.request.urlopen

        def no_network(*a, **k):
            calls.append(a)
            raise AssertionError('network')

        cd.urllib.request.urlopen = no_network
        try:
            os.environ['NTFY_TOPIC'] = 'x'
            cd.DRY_RUN = True
            cd.notify('t', 'm')
            self.assertEqual(calls, [])
        finally:
            cd.DRY_RUN = False
            cd.urllib.request.urlopen = orig
            os.environ.pop('NTFY_TOPIC', None)


class SymptomTable(unittest.TestCase):
    def test_add_and_describe(self):
        sym = cd.Symptoms()
        sym.add('zero_fps', 'restream up but Frigate reads 0 fps: backyard', ['backyard'])
        sym.add('frozen', 'latest.jpg unchanged', ['backyard', 'front_door'])
        self.assertEqual(sym.per_camera, {'backyard': ['zero_fps', 'frozen'], 'front_door': ['frozen']})
        self.assertEqual(sorted(sym.general), ['frozen', 'zero_fps'])
        self.assertEqual((sym.immediate, sym.dead), ({}, {}))
        self.assertEqual(sym.describe(['backyard', 'front_door'], LINK_TODAY),
                         'backyard: zero_fps+frozen [ping avg RTT 189 ms, 1 go2rtc read timeouts since last run]; '
                         'front_door: frozen')


# ---------------------------------------------------------------------------
# End-to-end replays with the outside world stubbed. The script looks these names up at call time, so
# patching the module attributes (and the shared os/subprocess/time modules) reaches it.
# ---------------------------------------------------------------------------
ORIG = {n: getattr(cd, n) for n in ('http_get', 'container_state', 'nfs_write_probe', 'notify', 'frame_hash',
                                     'probe_pids', 'probe_rss', 'STATE_FILE')}
ORIG_NOTIFY = cd.notify
ORIG_ISMOUNT = os.path.ismount
ORIG_RUN = subprocess.run
ORIG_SLEEP = time.sleep


class Fake:
    """Frigate, docker and ntfy as frigate_watchdog() sees them."""

    def __init__(self, fps, log_lines=(), stats_ok=True):
        self.fps, self.log_lines, self.stats_ok = fps, list(log_lines), stats_ok
        self.pushes, self.restarts, self.hashes = [], [], {}
        self.pid_alive, self.rss, self.pids = {}, {}, {}
        self.disabled = ()  # cameras switched off in Frigate's config
        self.calls = 0

    def fake_frame_hash(self, cam):
        self.calls += 1
        return self.hashes.get(cam, f'h-{cam}-{self.calls}')  # moves every call unless pinned

    def http_get(self, url, timeout=10):
        if '/api/stats' in url:
            if not self.stats_ok:
                raise OSError('down')
            cameras = {c: dict({'camera_fps': f}, **self.pids.get(c, {})) for c, f in self.fps.items()}
            return json.dumps({'cameras': cameras}).encode()
        if '/api/logs/frigate' in url:
            return json.dumps({'lines': self.log_lines}).encode()
        if '/api/config' in url:
            return json.dumps({'cameras': {c: {'enabled': c not in self.disabled} for c in cd.CAMERAS}}).encode()
        raise AssertionError(url)

    def fake_run(self, cmd, **_):
        self.restarts.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    def install(self, tmpdir):
        cd.http_get = self.http_get
        cd.container_state = lambda: ('running', NOW - 3 * 3600)
        cd.nfs_write_probe = lambda d, timeout=20: 'ok'
        cd.notify = lambda t, m: self.pushes.append((t, m))
        cd.frame_hash = self.fake_frame_hash
        cd.probe_pids = lambda pids: {int(p): self.pid_alive.get(int(p), True) for p in pids}
        cd.probe_rss = lambda pids: dict(self.rss)
        cd.STATE_FILE = Path(tmpdir) / 'state.json'
        os.path.ismount = lambda p: True
        subprocess.run = self.fake_run
        time.sleep = lambda s: None

    @staticmethod
    def uninstall():
        for n, f in ORIG.items():
            setattr(cd, n, f)
        os.path.ismount = ORIG_ISMOUNT
        subprocess.run = ORIG_RUN
        time.sleep = ORIG_SLEEP


class ReplayCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        Fake.uninstall()
        self.tmp.cleanup()

    def run_wd(self, fake, state, link, runs, now=NOW):
        fake.install(self.tmp.name)
        for i in range(runs):
            cd.frigate_watchdog(state, ALL_UP, now + 300 * i, False, link)
        return state


class WatchdogReplays(ReplayCase):
    def test_zero_fps_on_a_degraded_link_no_restart_no_push(self):
        # 17:10/17:15: the pre-attribution watchdog restarted Frigate for front_garage's 0 fps
        fake = Fake({'front_door': 5.0, 'front_garage': 0, 'backyard': 3.1, 'backyard_gate': 3.1})
        state = real_state()
        self.run_wd(fake, state, LINK_TODAY, runs=4)
        self.assertEqual(fake.restarts, [])
        self.assertEqual(fake.pushes, [])
        self.assertEqual(state['frigate']['strikes'], 0)
        self.assertEqual(state['frigate']['attribution'], {'front_garage': 'link'})

    def test_frozen_frames_on_degraded_wifi_cams_no_restart(self):
        # 03:05/03:10: camera_fps stays > 0 (a stale value) while latest.jpg freezes on the Tapo cameras
        fake = Fake({'front_door': 5.0, 'front_garage': 1.7, 'backyard': 3.0, 'backyard_gate': 3.0})
        fake.hashes.update(front_garage='same', backyard='same', backyard_gate='same')
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_TODAY, runs=4)  # seed + 3 frozen runs
        self.assertEqual(fake.restarts, [])
        self.assertEqual(fake.pushes, [])
        self.assertEqual(state['frigate']['strikes'], 0)
        self.assertEqual(state['frigate']['attribution'],
                         {'front_garage': 'link', 'backyard': 'link', 'backyard_gate': 'link'})

    def test_pipeline_wide_zero_fps_restarts_after_two_runs(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 0))
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=1)
        self.assertEqual(fake.restarts, [])
        self.assertEqual(state['frigate']['strikes'], 1)
        self.run_wd(fake, state, LINK_CLEAN, runs=1, now=NOW + 300)
        self.assertEqual(fake.restarts, [[cd.DOCKER, 'restart', cd.FRIGATE_CONTAINER]])
        self.assertEqual(state['frigate']['strikes'], 0)
        self.assertTrue(state['frigate']['verify_after_restart'])
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: Frigate auto-restarted'])
        self.assertIn('Do:', fake.pushes[0][1])
        # the next clean run sends exactly one "healthy again"
        fake.fps = dict.fromkeys(cd.CAMERAS, 5.0)
        self.run_wd(fake, state, LINK_CLEAN, runs=1, now=NOW + 900)
        self.assertEqual([t for t, _ in fake.pushes][-1], 'VisionBox: Frigate healthy again')
        self.assertNotIn('verify_after_restart', state['frigate'])

    def test_wired_zero_fps_with_wifi_link_noise_restarts(self):
        fake = Fake({'front_door': 0, 'front_garage': 0, 'backyard': 3.0, 'backyard_gate': 3.0})
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_TODAY, runs=2)
        self.assertEqual(len(fake.restarts), 1)
        self.assertEqual(state['frigate']['attribution'], {'front_door': 'pipeline', 'front_garage': 'link'})

    def test_dead_thread_restarts_immediately(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 5.0), log_lines=[
            '2026-09-13 18:54:00.0  OSError: [Errno 116] Stale file handle',
            '2026-09-13 18:54:01.0  Exception in thread event_cleanup:'])
        state = cd.normalise_state({})
        state['frigate']['log_cursor'] = NOW - 600
        self.run_wd(fake, state, LINK_TODAY, runs=1)
        self.assertEqual(len(fake.restarts), 1)
        titles = [t for t, _ in fake.pushes]
        self.assertIn('VisionBox: NFS stale file handles', titles)
        self.assertIn('VisionBox: Frigate auto-restarted', titles)

    def test_budget_blocks_and_human_alert_once_per_6h(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 0))
        state = real_state()  # three restarts in the last 6 h
        state['frigate']['alerts'] = {}
        self.run_wd(fake, state, LINK_CLEAN, runs=8)  # 40 min of pipeline-wide 0 fps
        self.assertEqual(fake.restarts, [])
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: Frigate keeps failing'])
        self.assertIn('budget', fake.pushes[0][1])
        self.run_wd(fake, state, LINK_CLEAN, runs=1, now=NOW + 6 * 3600 + 400)  # the window slid: restart
        self.assertEqual(len(fake.restarts), 1)

    def test_cooldown_blocks(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 0))
        state = cd.normalise_state({})
        state['frigate']['restarts'] = [NOW - 600]
        self.run_wd(fake, state, LINK_CLEAN, runs=2)
        self.assertEqual(fake.restarts, [])
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: Frigate keeps failing'])
        self.assertIn('cooldown', fake.pushes[0][1])

    def test_single_wifi_pipeline_cam_alerts_human_not_restart(self):
        fake = Fake({'front_door': 5.0, 'front_garage': 5.0, 'backyard': 0, 'backyard_gate': 5.0})
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=10)
        self.assertEqual(fake.restarts, [])
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: Frigate keeps failing'])
        self.assertIn('single WiFi camera', fake.pushes[0][1])

    def test_api_down_is_a_pipeline_strike_restart_on_second_run(self):
        fake = Fake({}, stats_ok=False)
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=1)
        self.assertEqual(fake.restarts, [])  # first run: strike only
        self.assertEqual(state['frigate']['strikes'], 1)
        self.assertEqual(state['frigate']['attribution'], {'front_door': 'pipeline'})
        self.run_wd(fake, state, LINK_CLEAN, runs=1, now=NOW + 300)
        self.assertEqual(len(fake.restarts), 1)  # second consecutive run: restart

    def test_api_down_counts_even_when_wifi_links_are_bad(self):
        fake = Fake({}, stats_ok=False)
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_TODAY, runs=2)
        self.assertEqual(len(fake.restarts), 1)

    def test_camera_disabled_in_frigate_is_ignored(self):
        fake = Fake({'front_door': 0, 'front_garage': 5.0, 'backyard': 5.0, 'backyard_gate': 5.0})
        fake.disabled = ('front_door',)
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=3)
        self.assertEqual((fake.restarts, state['frigate']['strikes'], state['frigate']['attribution']), ([], 0, {}))

    def test_api_down_once_then_recovered_never_restarts(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 5.0), stats_ok=False)
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=1)
        fake.stats_ok = True
        self.run_wd(fake, state, LINK_CLEAN, runs=3, now=NOW + 300)
        self.assertEqual(fake.restarts, [])
        self.assertEqual(state['frigate']['strikes'], 0)

    def test_dry_run_never_restarts(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 0))
        state = cd.normalise_state({})
        cd.DRY_RUN = True
        try:
            self.run_wd(fake, state, LINK_CLEAN, runs=3)
        finally:
            cd.DRY_RUN = False
        self.assertEqual(fake.restarts, [])
        self.assertEqual(state['frigate']['restarts'], [])


class WatchdogGates(ReplayCase):
    def test_container_not_running_resets_strikes_and_alerts_on_second_run(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 5.0))
        fake.install(self.tmp.name)
        cd.container_state = lambda: ('exited', 0.0)
        state = cd.normalise_state({})
        state['frigate']['strikes'] = 1
        self.assertFalse(cd.container_ready(state, NOW, False))
        self.assertEqual((state['frigate']['down_runs'], state['frigate']['strikes']), (1, 0))
        self.assertEqual(fake.pushes, [])
        self.assertFalse(cd.container_ready(state, NOW + 300, False))
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: Frigate container not running'])

    def test_warmup_is_left_alone(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 5.0))
        fake.install(self.tmp.name)
        state = cd.normalise_state({})
        state['frigate'].update(strikes=2, frame_hash={'backyard': 'h'}, down_runs=3)
        cd.container_state = lambda: ('running', NOW - 60)
        self.assertFalse(cd.container_ready(state, NOW, False))
        self.assertEqual((state['frigate']['strikes'], state['frigate']['frame_hash']), (0, {}))
        self.assertEqual(state['frigate']['down_runs'], 0)
        cd.container_state = lambda: ('running', NOW - 3600)
        self.assertTrue(cd.container_ready(state, NOW, False))

    def test_fetch_stats_retries_once(self):
        fake = Fake({'front_door': 5.0}, stats_ok=False)
        fake.install(self.tmp.name)
        self.assertEqual(cd.fetch_stats(), (None, 'OSError'))
        timeouts = []

        def flaky(url, timeout=10):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise OSError('down')
            return fake.http_get(url, timeout)

        fake.stats_ok = True
        cd.http_get = flaky
        cams, err = cd.fetch_stats()
        self.assertEqual((cams['front_door']['camera_fps'], err, timeouts), (5.0, '', [10, 15]))


class RebootGating(unittest.TestCase):
    def test_unmeasured_or_stale_keeps_old_behaviour(self):
        self.assertEqual(cd.reboot_allowed({}, NOW), (True, ''))
        self.assertEqual(cd.reboot_allowed({'at': NOW - 300, 'loss_pct': 100.0}, NOW), (True, ''))
        self.assertEqual(cd.reboot_allowed({'at': NOW, 'loss_pct': None}, NOW), (True, ''))

    def test_hung_firmware_signature_allows_reboot(self):
        self.assertTrue(cd.reboot_allowed({'at': NOW, 'loss_pct': 0.0, 'rtt_avg_ms': 3.2}, NOW)[0])

    def test_unreachable_camera_is_not_rebooted(self):
        ok, why = cd.reboot_allowed({'at': NOW, 'loss_pct': 100.0, 'rtt_avg_ms': None}, NOW)
        self.assertFalse(ok)
        self.assertIn('unreachable', why)

    def test_lossy_or_slow_link_is_not_rebooted(self):
        ok, why = cd.reboot_allowed({'at': NOW, 'loss_pct': 40.0, 'rtt_avg_ms': 90.0}, NOW)
        self.assertFalse(ok)
        self.assertIn('radio', why)
        ok, why = cd.reboot_allowed({'at': NOW, 'loss_pct': 0.0, 'rtt_avg_ms': 400.0}, NOW)
        self.assertFalse(ok)

    def test_link_alert_repeat_is_six_hours(self):
        self.assertGreaterEqual(cd.LINK_ALERT_REPEAT_S, 6 * 3600)
        self.assertEqual(cd.PING_COUNT, 10)


URL_TO_CAM = {c['rtsp']: n for n, c in cd.CAMERAS.items()}
RESTREAM_TO_CAM = {c['restream']: n for n, c in cd.CAMERAS.items()}


class CameraLayer(unittest.TestCase):
    """probe_cameras(), reboot_cameras() and evaluate_links() with the probes stubbed."""

    def setUp(self):
        self.alive = dict.fromkeys(cd.CAMERAS, True)
        self.restream_status = dict.fromkeys(cd.CAMERAS, 200)
        self.pushes, self.reboots = [], []
        self.orig = {n: getattr(cd, n) for n in ('rtsp_alive', 'rtsp_status', 'notify', 'onvif_reboot',
                                                  'measure_links')}
        cd.rtsp_alive = lambda url, timeout=4.0: self.alive[URL_TO_CAM[url]]
        cd.rtsp_status = lambda url, timeout=4.0: self.restream_status[RESTREAM_TO_CAM[url.rsplit('/', 1)[1]]]
        cd.notify = lambda t, m: self.pushes.append((t, m))
        cd.onvif_reboot = lambda host, *a: self.reboots.append(host) or 'accepted'

    def tearDown(self):
        for n, f in self.orig.items():
            setattr(cd, n, f)

    def probe(self, state, runs, now=NOW):
        for i in range(runs):
            res = cd.probe_cameras(state, now + 300 * i, False)
        return res

    def test_down_after_three_misses_then_recovery(self):
        state = cd.normalise_state({})
        self.alive['front_garage'] = False
        alive, restream_up, reboot = self.probe(state, 2)
        self.assertEqual((alive['front_garage'], state['front_garage']['fails'], reboot), (False, 2, []))
        self.assertNotIn('front_garage', restream_up)  # no restream verdict while the camera itself is down
        self.assertEqual(self.pushes, [])
        _, _, reboot = self.probe(state, 1, NOW + 600)
        self.assertEqual(reboot, ['front_garage'])
        self.assertEqual([t for t, _ in self.pushes], ['VisionBox: front_garage DOWN'])
        self.assertTrue(state['front_garage']['notified'])
        self.assertEqual(state['front_garage']['down_since'], NOW + 600 - 3 * 300)
        self.alive['front_garage'] = True
        _, restream_up, reboot = self.probe(state, 1, NOW + 900)
        self.assertEqual((reboot, [t for t, _ in self.pushes][-1]), ([], 'VisionBox: front_garage recovered'))
        self.assertEqual((state['front_garage']['fails'], state['front_garage']['down_since']), (0, None))
        self.assertTrue(restream_up['front_garage'])

    def test_reboot_wanted_respects_the_hourly_cooldown(self):
        state = cd.normalise_state({})
        state['backyard']['last_reboot'] = NOW - 600
        self.alive['backyard'] = False
        _, _, reboot = self.probe(state, 3)
        self.assertEqual(reboot, [])
        _, _, reboot = self.probe(state, 1, NOW + 3600)
        self.assertEqual(reboot, ['backyard'])

    def test_restream_judged_only_while_the_camera_is_up(self):
        state = cd.normalise_state({})
        self.restream_status['backyard'] = 404  # go2rtc has no producer
        _, restream_up, _ = self.probe(state, 2)
        self.assertFalse(restream_up['backyard'])
        self.assertEqual(state['backyard']['restream_fails'], 2)
        self.alive['backyard'] = False
        _, restream_up, _ = self.probe(state, 1, NOW + 600)
        self.assertNotIn('backyard', restream_up)
        self.assertEqual(state['backyard']['restream_fails'], 2)
        self.alive['backyard'] = True
        self.restream_status['backyard'] = 401  # asks for auth: still serving
        _, restream_up, _ = self.probe(state, 1, NOW + 900)
        self.assertTrue(restream_up['backyard'])
        self.assertEqual(state['backyard']['restream_fails'], 0)

    def test_reboot_cameras_gates_on_the_link_and_dry_run(self):
        state = cd.normalise_state({})
        state['front_garage']['link']['last'] = {'at': NOW, 'loss_pct': 100.0, 'rtt_avg_ms': None}
        state['backyard']['link']['last'] = {'at': NOW, 'loss_pct': 0.0, 'rtt_avg_ms': 3.0}
        cd.reboot_cameras(state, ['front_garage', 'backyard'], NOW)
        self.assertEqual(self.reboots, ['192.168.1.101'])
        self.assertEqual(state['backyard']['last_reboot'], NOW)
        self.assertEqual((state['front_garage']['last_reboot'], state['front_garage']['reboot_skipped_at']), (0, NOW))
        cd.DRY_RUN = True
        try:
            cd.reboot_cameras(state, ['backyard_gate'], NOW)
        finally:
            cd.DRY_RUN = False
        self.assertEqual(self.reboots, ['192.168.1.101'])
        self.assertEqual(state['backyard_gate']['last_reboot'], 0)

    def test_evaluate_links_pushes_degraded_once_then_recovered(self):
        state = cd.normalise_state({})
        link = dict(LINK_CLEAN, backyard=LINK_TODAY['backyard'])
        cd.measure_links = lambda *a: link
        state['backyard']['link']['bad_since'] = NOW - 900
        state['backyard']['link']['last'] = {'loss_pct': 0.0, 'rtt_avg_ms': 188.9, 'rtt_max_ms': 504.1,
                                             'producer_sampled': True, 'producer_present': True,
                                             'producer_bps': 50000.0, 'go2rtc_timeouts': 1}
        self.assertIs(cd.evaluate_links(state, ALL_UP, NOW, False), link)
        self.assertEqual([t for t, _ in self.pushes], ['VisionBox: backyard link degraded'])
        self.assertIn('avg RTT 189 ms', self.pushes[0][1])
        self.assertEqual((state['backyard']['link']['status'], state['backyard']['link']['alerted']),
                         ('degraded', True))
        cd.evaluate_links(state, ALL_UP, NOW + 300, False)  # still degraded: no second push
        self.assertEqual(len(self.pushes), 1)
        cd.measure_links = lambda *a: LINK_CLEAN
        cd.evaluate_links(state, ALL_UP, NOW + 600, False)
        cd.evaluate_links(state, ALL_UP, NOW + 600 + 900, False)
        self.assertEqual([t for t, _ in self.pushes][-1], 'VisionBox: backyard link recovered')
        self.assertEqual((state['backyard']['link']['status'], state['backyard']['link']['alerted']), ('ok', False))

    def test_dead_restream_blames_frigate_only_on_a_clean_link(self):
        state = cd.normalise_state({})
        state['backyard']['restream_fails'] = cd.FAILS_BEFORE_DOWN
        state['front_garage']['restream_fails'] = cd.FAILS_BEFORE_DOWN
        cd.measure_links = lambda *a: dict(LINK_TODAY, backyard=OK_LINK)
        cd.evaluate_links(state, ALL_UP, NOW, False)
        self.assertEqual([t for t, _ in self.pushes], ['VisionBox: restream problem'])
        self.assertIn('backyard camera and its link are fine', self.pushes[0][1])

    def test_evaluate_links_survives_a_measurement_error(self):
        def boom(*a):
            raise RuntimeError('ping missing')

        cd.measure_links = boom
        self.assertEqual(cd.evaluate_links(cd.normalise_state({}), ALL_UP, NOW, False), {})
        self.assertEqual(self.pushes, [])


class DeadProcesses(ReplayCase):
    def test_dead_capture_is_flagged(self):
        alive = {1187: False, 1084: True, 894: True, 837: True}
        self.assertEqual(cd.dead_processes(PID_STATS, alive, ['front_garage', 'backyard']),
                         {'front_garage': ['capture']})

    def test_alive_or_unknown_is_not_flagged(self):
        self.assertEqual(cd.dead_processes(PID_STATS, {1187: True, 1084: True}, ['front_garage', 'backyard']), {})
        self.assertEqual(cd.dead_processes(PID_STATS, {}, ['front_garage']), {})
        self.assertEqual(cd.dead_processes({'front_garage': {}}, {1: False}, ['front_garage']), {})

    def test_solo_camera_with_dead_process_restarts_after_two_runs(self):
        r, s = cd.decide({}, {'front_garage': 'pipeline'}, 0, solo=('front_garage',))
        self.assertEqual((r, s), (False, 1))
        r, s = cd.decide({}, {'front_garage': 'pipeline'}, 1, solo=('front_garage',))
        self.assertEqual((r, s), (True, 2))
        r, s = cd.decide({}, {'front_garage': 'pipeline'}, 1)  # not solo: a single WiFi camera never restarts
        self.assertEqual((r, s), (False, 2))

    def test_oom_killed_capture_on_a_degraded_link_restarts(self):
        """Replay of 2026-09-14 00:26 UTC: front_garage's capture process was OOM-killed while its link was
        degraded, and the frozen frame was filed under 'link' for 30 min. A dead process is always pipeline."""
        fake = Fake({'front_door': 5.0, 'front_garage': 104.5, 'backyard': 3.0, 'backyard_gate': 3.0})
        fake.pids = {'front_garage': {'capture_pid': 1187, 'pid': 1084}}
        fake.pid_alive = {1187: False, 1084: True}
        fake.hashes['front_garage'] = 'same'
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_TODAY, runs=1)
        self.assertEqual(fake.restarts, [])
        self.assertEqual(state['frigate']['attribution'], {'front_garage': 'pipeline'})
        self.run_wd(fake, state, LINK_TODAY, runs=1, now=NOW + 300)
        self.assertEqual(len(fake.restarts), 1)
        self.assertIn('process_dead', fake.pushes[-1][1])

    def test_alive_processes_change_nothing(self):
        fake = Fake({'front_door': 5.0, 'front_garage': 0, 'backyard': 3.0, 'backyard_gate': 3.0})
        fake.pids = {c: {'capture_pid': 100 + i, 'pid': 200 + i} for i, c in enumerate(cd.CAMERAS)}
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_TODAY, runs=3)
        self.assertEqual(fake.restarts, [])

    def test_capture_rss_balloon_alerts_once(self):
        fake = Fake(dict.fromkeys(cd.CAMERAS, 5.0))
        fake.pids = {'backyard': {'capture_pid': 894, 'pid': 837}}
        fake.rss = {894: 2100.0}
        state = cd.normalise_state({})
        self.run_wd(fake, state, LINK_CLEAN, runs=3)
        self.assertEqual([t for t, _ in fake.pushes], ['VisionBox: backyard capture process ballooning'])
        self.assertEqual(fake.restarts, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
