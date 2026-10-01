"""MP4 rendering: title card, then the main figure with a moving playhead, with the audio.

ffmpeg is taken from ``$PATH`` or, failing that, from the ``imageio-ffmpeg`` package
(``pip install "gwsonify[video]"``). Without either, :func:`find_ffmpeg` raises
:class:`~gwsonify.MissingDependencyError` and the pipeline skips the video with a
message.

The playhead position is computed from the audio time: an audio time ``tau`` shows
data time ``window_start + tau / duration_factor``, so the line stays in sync under
``--speed`` and ``--stretch``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gwsonify import GwsonifyError, MissingDependencyError
from gwsonify.audio import AUDIO_RATE, write_wav

log = logging.getLogger(__name__)

FPS = 25
TITLE_SECONDS = 3.0


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise MissingDependencyError(
            "Video needs ffmpeg: install it (e.g. `brew install ffmpeg`, `conda install ffmpeg`, "
            "`apt install ffmpeg`) or `pip install \"gwsonify[video]\"`."
        ) from None


def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render(path: Path, fig, axes, audio: np.ndarray, window: tuple[float, float],
           duration_factor: float, title_fig=None) -> Path:
    """Write an H.264/AAC MP4 of ``fig`` with a playhead on ``axes`` and ``audio``.

    The figure is rendered once. Each frame copies that image and paints the playhead
    columns directly, which keeps long (BNS-length) videos fast.
    """
    import matplotlib.pyplot as plt

    ffmpeg = find_ffmpeg()
    path = Path(path)
    background = _rgb(fig)
    full_h = background.shape[0]
    height, width = full_h - full_h % 2, background.shape[1] - background.shape[1] % 2
    background = np.ascontiguousarray(background[:height, :width])
    # pixel geometry of each axes (display coordinates have their origin bottom-left)
    spans = []
    for i, ax in enumerate(axes):
        bb = ax.get_window_extent()
        rows = slice(max(int(full_h - bb.y1), 0), min(int(full_h - bb.y0), height))
        color = np.array([255, 255, 255] if i % 2 == 0 else [0, 0, 0], dtype=np.uint8)
        spans.append((ax, rows, color, int(bb.x0), int(bb.x1)))

    title_frames = 0
    title_rgb = None
    if title_fig is not None:
        title_rgb = np.ascontiguousarray(_rgb(title_fig)[:height, :width]).tobytes()
        title_frames = int(TITLE_SECONDS * FPS)
        plt.close(title_fig)

    audio_seconds = len(audio) / AUDIO_RATE
    n_frames = int(np.ceil(audio_seconds * FPS)) + 1
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        lead = np.zeros((title_frames * AUDIO_RATE // FPS, *audio.shape[1:]))
        write_wav(wav, np.concatenate([lead, audio]))
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-i", str(wav),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
               str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        frame = background.copy()
        painted: list[tuple[slice, int]] = []
        try:
            for _ in range(title_frames):
                proc.stdin.write(title_rgb)
            for k in range(n_frames):
                t = window[0] + (k / FPS) / duration_factor
                for rows, x in painted:  # undo the previous playhead
                    frame[rows, x : x + 2] = background[rows, x : x + 2]
                painted = []
                for ax, rows, color, x0, x1 in spans:
                    x = int(round(ax.transData.transform((t, 0.0))[0]))
                    if x0 <= x < x1 - 1:
                        frame[rows, x : x + 2] = color
                        painted.append((rows, x))
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise GwsonifyError(f"ffmpeg failed: {err.strip()[:500]}")
    return path
