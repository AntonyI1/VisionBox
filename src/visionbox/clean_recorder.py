"""FFmpeg clean recorder — copies H.264 from RTSP with zero CPU."""

import json
import signal
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone


class CleanRecorder:
    """Records raw RTSP stream via FFmpeg -c copy. No bounding boxes."""

    def __init__(self, output_dir: str | Path, rtsp_url: str = ''):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rtsp_url = rtsp_url
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
        self._start_time = datetime.now(timezone.utc)
        self._current_path = self.output_dir / f"event_{event_id}.mp4"

        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c', 'copy', '-an',
            '-movflags', '+faststart',
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
            self._process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            self._process.kill()
            self._process.wait()

        self._process = None
        path = self._current_path

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
