"""FFmpeg clean recorder: copies H.264 from RTSP with zero CPU."""

import json
import logging
import shutil
import signal
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ffmpeg exits 255 on the SIGINT we send; the -t cap ends with 0.
_CLEAN_EXIT_CODES = (0, 255)
_STDERR_TAIL = 400


class CleanRecorder:
    """Records raw RTSP stream via FFmpeg -c copy. No bounding boxes."""

    def __init__(self, output_dir: str | Path, rtsp_url: str = '',
                 max_duration: float = 130.0, min_valid_bytes: int = 51200,
                 stop_grace: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rtsp_url = rtsp_url
        self.max_duration = max_duration
        self.min_valid_bytes = min_valid_bytes
        self.stop_grace = stop_grace
        self._process: subprocess.Popen | None = None
        self._current_path: Path | None = None
        self._stderr_path: Path | None = None
        self._event_id: str | None = None
        self._start_time: datetime | None = None
        self._available = bool(
            shutil.which('ffmpeg') and rtsp_url and rtsp_url.startswith('rtsp://')
        )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def event_id(self) -> str | None:
        """Event whose clip was started and not yet finalised, even if ffmpeg has exited."""
        return self._event_id

    def start_event(self, event_id: str) -> str | None:
        if not self._available or self.is_recording:
            return None

        path = self.output_dir / f'event_{event_id}.mp4'
        stderr_path = self.output_dir / f'.event_{event_id}.ffmpeg.log'
        cmd = [
            'ffmpeg', '-nostdin', '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c', 'copy', '-an',
            '-t', str(self.max_duration),
            # Fragmented MP4: a SIGKILL/-t cut still leaves a playable file (no trailing moov needed).
            '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            '-y', str(path),
        ]

        try:
            with open(stderr_path, 'wb') as stderr:
                self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr)
        except OSError as exc:
            logger.warning('clean recorder for %s could not start ffmpeg: %s', event_id, exc)
            self._process = None
            with suppress(OSError):
                stderr_path.unlink()
            return None

        self._event_id = event_id
        self._start_time = datetime.now(UTC)
        self._current_path = path
        self._stderr_path = stderr_path
        return str(path)

    def stop_event(self) -> str | None:
        if self._event_id is None:
            return None

        killed = False
        returncode = None
        if self._process is not None:
            try:
                self._process.send_signal(signal.SIGINT)   # no-op if ffmpeg already exited
                returncode = self._process.wait(timeout=self.stop_grace)
            except (subprocess.TimeoutExpired, OSError):
                self._process.kill()
                returncode = self._process.wait()
                killed = True
            self._process = None

        event_id, path = self._event_id, self._current_path
        size = path.stat().st_size if path and path.exists() else -1
        problem = (size < self.min_valid_bytes) or (not killed and returncode not in _CLEAN_EXIT_CODES)
        if problem:
            logger.warning('clean clip for %s: ffmpeg exit %s, %s bytes%s',
                           event_id, returncode, max(size, 0), self._stderr_tail())
        if self._stderr_path:
            with suppress(OSError):
                self._stderr_path.unlink()

        # Fragmented MP4 means even a killed ffmpeg leaves a playable file, so anything under
        # the size floor is genuinely broken; return None so the caller clears the DB column.
        if path and 0 <= size < self.min_valid_bytes:
            with suppress(OSError):
                path.unlink()
        if size < self.min_valid_bytes:
            path = None

        if path and self._start_time:
            meta_path = path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'event_id': event_id,
                    'start_time': self._start_time.isoformat(),
                    'end_time': datetime.now(UTC).isoformat(),
                    'type': 'clean',
                    'source': self.rtsp_url,
                }, f, indent=2)

        self._current_path = None
        self._stderr_path = None
        self._event_id = None
        self._start_time = None
        return str(path) if path else None

    def _stderr_tail(self) -> str:
        if not self._stderr_path:
            return ''
        try:
            text = self._stderr_path.read_text(errors='replace').strip()
        except OSError:
            return ''
        return f': {text[-_STDERR_TAIL:]}' if text else ''

    def release(self):
        if self._event_id is not None:
            self.stop_event()
