"""Stage 4 video utilities (CPU-only, OpenCV)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def _sorted_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {input_dir}/{pattern}")
    return files


def images_to_video(
    input_dir: str,
    output_path: str,
    fps: int = 1,
    pattern: str = "frame_*_redesigned.png",
):
    """Compile images into an MP4 video using OpenCV VideoWriter."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for video output.") from exc

    in_dir = Path(input_dir)
    out_path = Path(output_path)

    files = _sorted_files(in_dir, pattern)

    first = cv2.imread(str(files[0]))
    if first is None:
        raise RuntimeError(f"Failed to read image: {files[0]}")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    try:
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {f}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    return str(out_path)
