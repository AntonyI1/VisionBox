"""FFmpeg clean recorder — copies H.264 from RTSP with zero CPU."""

import json
import shutil
import signal
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path


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

    def start_event(self, event_id: str) -> str | None:
        if not self._available or self.is_recording:
            return None

        self._event_id = event_id
        self._start_time = datetime.now(UTC)
        self._current_path = self.output_dir / f"event_{event_id}.mp4"

        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c', 'copy', '-an',
            '-t', str(self.max_duration),
            # Fragmented MP4: a SIGKILL/-t cut still leaves a playable file (no trailing moov needed).
            '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            '-y', str(self._current_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            self._process = None
            return None

        return str(self._current_path)

    def stop_event(self) -> str | None:
        if self._process is None:
            return None

        try:
            self._process.send_signal(signal.SIGINT)
            self._process.wait(timeout=self.stop_grace)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait()

        self._process = None
        path = self._current_path

        # Fragmented MP4 means even a killed ffmpeg leaves a playable file, so anything under
        # the size floor is genuinely broken; return None so the caller clears the DB column.
        if path and path.exists() and path.stat().st_size < self.min_valid_bytes:
            with suppress(OSError):
                path.unlink()
            path = None

        if path and path.exists() and self._start_time:
            meta_path = path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'event_id': self._event_id,
                    'start_time': self._start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'type': 'clean',
                    'source': self.rtsp_url,
                }, f, indent=2)

        self._current_path = None
        self._event_id = None
        self._start_time = None
        return str(path) if path else None

    def release(self):
        if self.is_recording:
            self.stop_event()
