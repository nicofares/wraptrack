"""
Convert a stack of images into an MP4 video.

Supports:
  - A single multi-page/animated TIFF file
  - A single PNG file (produces a short static video)
  - A folder containing a stack of single-frame images (.tif/.tiff/.png,
    can be mixed) — one file per frame

Usage:
    python images_to_mp4.py input.tiff output.mp4 --fps 10
    python images_to_mp4.py input.png output.mp4 --fps 10
    python images_to_mp4.py /path/to/folder/ output.mp4 --fps 10

Requirements:
    pip install tifffile opencv-python numpy imageio --break-system-packages
"""

import argparse
import re
import sys
import glob
from pathlib import Path

import numpy as np
import cv2
import tifffile
import imageio.v3 as iio


def _load_frames(tiff_path):
    """Load all pages/frames from a multi-page TIFF file as a list of numpy arrays."""
    data = tifffile.imread(tiff_path)

    if data.ndim == 2:
        frames = [data]
    elif data.ndim == 3:
        if data.shape[-1] in (1, 3, 4):
            frames = [data]
        else:
            frames = [data[i] for i in range(data.shape[0])]
    elif data.ndim == 4:
        frames = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError(f"Unsupported TIFF array shape: {data.shape}")

    return frames


def _load_single_image(path):
    """Load one single-frame image file (TIFF page or PNG) as a numpy array."""
    suffix = Path(path).suffix.lower()
    if suffix in ('.tif', '.tiff'):
        return tifffile.imread(path)
    elif suffix == '.png':
        return iio.imread(path)
    else:
        raise ValueError(f"Unsupported image extension: {path}")


def natural_sort_key(path):
    """Sort key that orders filenames 'naturally' (frame_2 before frame_10)."""
    name = Path(path).name
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r'(\d+)', name)]


def load_frames(path):
    """Load frames from a path that may be a folder of single-frame images,
    a single multi-page TIFF file, or a single PNG file."""
    p = Path(path)

    if p.is_dir():
        patterns = ('*.tif', '*.tiff', '*.png', '*.TIF', '*.TIFF', '*.PNG')
        filepaths = []
        for pattern in patterns:
            filepaths.extend(glob.glob(str(p / pattern)))
        filepaths = sorted(set(filepaths), key=natural_sort_key)

        if not filepaths:
            raise FileNotFoundError(f"No .tif/.tiff/.png files found in folder: {path}")

        print(filepaths[:20])
        return [_load_single_image(fp) for fp in filepaths]

    elif p.is_file():
        suffix = p.suffix.lower()
        if suffix in ('.tif', '.tiff'):
            return _load_frames(str(p))
        elif suffix == '.png':
            return [_load_single_image(str(p))]
        else:
            raise ValueError(f"Unsupported file type: {path}")

    else:
        raise FileNotFoundError(f"Path does not exist: {path}")


def compute_global_range(frames):
    """Compute one shared (min, max) used to rescale every frame, keeping
    brightness consistent across the whole video instead of flickering
    frame-to-frame. Sampled from the first 10 frames by design."""
    fmin = min(np.asarray(f).min() for f in frames[:10])
    fmax = max(np.asarray(f).max() for f in frames[:10])
    return float(fmin), float(fmax)


def normalize_frame(frame, global_range=None):
    """Convert a frame to 8-bit BGR for video writing."""
    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if global_range is not None:
            fmin, fmax = global_range
        else:
            fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            frame = (frame - fmin) / (fmax - fmin) * 255.0
            frame = np.clip(frame, 0, 255)
        else:
            frame = np.zeros_like(frame)
        frame = frame.astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3:
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported channel count: {frame.shape[-1]}")

    return frame


def convert(input_path, mp4_path, fps=10, debug=True):
    frames = load_frames(input_path)
    if not frames:
        raise ValueError("No frames found.")

    if debug:
        raw_mins = [float(np.asarray(f).min()) for f in frames]
        raw_maxs = [float(np.asarray(f).max()) for f in frames]
        # print(f"[debug] loaded {len(frames)} frame(s), dtype={np.asarray(frames[0]).dtype}, "
        #       f"shape={np.asarray(frames[0]).shape}")
        # print(f"[debug] per-frame min range: {min(raw_mins):.1f} to {max(raw_mins):.1f}")
        # print(f"[debug] per-frame max range: {min(raw_maxs):.1f} to {max(raw_maxs):.1f}")

    global_range = compute_global_range(frames)
    fmin, fmax = global_range
    if debug:
        # print(f"[debug] global_range used for normalization: ({fmin:.1f}, {fmax:.1f})")
        if fmax <= fmin:
            print(f"[debug] WARNING: global_range is degenerate (max <= min) -> ALL frames will render solid black")
        else:
            for i, f in enumerate(frames):
                arr = np.asarray(f)
                fr_min, fr_max = float(arr.min()), float(arr.max())
                if fr_max <= fmin:
                    print(f"[debug] WARNING: frame {i} true range ({fr_min:.1f}-{fr_max:.1f}) is "
                          f"entirely BELOW global_range min ({fmin:.1f}) -> this frame will render solid black")
                elif fr_min >= fmax:
                    print(f"[debug] WARNING: frame {i} true range ({fr_min:.1f}-{fr_max:.1f}) is "
                          f"entirely ABOVE global_range max ({fmax:.1f}) -> this frame will render solid white")

    frames = [normalize_frame(f, global_range) for f in frames]
    height, width = frames[0].shape[:2]

    if debug:
        post_mins = [int(f.min()) for f in frames]
        post_maxs = [int(f.max()) for f in frames]
        # print(f"[debug] after normalization, per-frame 8-bit min range: {min(post_mins)} to {max(post_mins)}")
        # print(f"[debug] after normalization, per-frame 8-bit max range: {min(post_maxs)} to {max(post_maxs)}")

    if len(frames) == 1:
        print("Note: only one frame/page found. "
              "The output video will just repeat that single frame.")

    # cv2.VideoWriter + mp4v: the same encoder from the original working script.
    # No ffmpeg, no lossless mode, no manual padding -- kept deliberately simple
    # and unchanged, since this is the one part of the pipeline confirmed to
    # actually produce a valid, non-black video.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Check codec/output path.")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    if len(frames) == 1:
        for _ in range(int(fps * 2) - 1):
            writer.write(frames[0])

    writer.release()
    print(f"Wrote {len(frames)} frame(s) at {fps} fps -> {mp4_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TIFF/PNG file or folder of images to an MP4 video."
    )
    parser.add_argument("input", help="Path to input .tif/.tiff/.png file, or a folder of images")
    parser.add_argument("output", help="Path to the output .mp4 file")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--no-debug", action="store_true",
                         help="Suppress the [debug] diagnostic prints about intensity ranges.")
    args = parser.parse_args()

    try:
        convert(args.input, args.output, fps=args.fps, debug=not args.no_debug)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()