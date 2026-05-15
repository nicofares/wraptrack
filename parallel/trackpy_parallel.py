# -*- coding: utf-8 -*-

"""
Tracking particles using Trackpy features, parallel computing.

Author: nicofares

Creation: 2026-05-15
"""

# To do:
# Add progress bar

# ===========================================================================================
# Importations
# ===========================================================================================
import numpy as np 
import matplotlib.pyplot as plt 

import trackpy as tp

import pims
import pandas as pd

import os 
import time 
from tqdm import tqdm

from multiprocessing import shared_memory, Pool
from functools import partial


# ===========================================================================================
# Functions
# ===========================================================================================


# ── 1. Loading ──────────────────────────────────────────────────────────────

def gray(image, channel=2):
    return image[:, :, channel]

def donothing(image):
    return image

def load_video_to_shared_memory(
        video_path: str, 
        start: int, stop: int, 
        bg: np.ndarray | pims.Frame.frame | pims.image_reader.ImageReader | None = None, 
        f: function = donothing,
    ):
    """
    Read all frames once into a shared memory block.
    """
    with pims.Video(video_path) as video:
        video = video[start:stop]
        n_frames = len(video)
        frame0 = np.asarray(gray(video[0]))
        shape = (n_frames, *frame0.shape)  # (T, H, W) or (T, H, W, C)
        dtype = frame0.dtype

    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    print(f"{nbytes / 1e9:.2f} GB will be loaded into shared memory.")
    shm = shared_memory.SharedMemory(create=True, size=nbytes)

    frames = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    with pims.Video(video_path) as video:
        video = video[start:stop]
        for i in tqdm(range(n_frames)):
            frame = video[i]
            # Background correction and image treatment
            if bg is not None:
                frame = frame - bg
            frame = f(frame)
            frames[i] = np.asarray(gray(frame))

    del frame0, frame 

    print(f"Loaded {n_frames} frames, {nbytes / 1e9:.2f} GB into shared memory.")
    return shm, shape, dtype


def release_shared_memory(shm: shared_memory.SharedMemory):
    shm.close()
    shm.unlink()


# ── 2. Per-frame work (queue-agnostic) ──────────────────────────────────────

def track_frame(i: int, frame: np.ndarray, args: tuple):
    """
    Trackpy logic
    """
    start, _, R1, minmass, separation, invert = args
    # Background correction (or any other modification) could also be added here 
    ftemp = tp.locate(frame, R1, invert=invert, minmass=minmass, separation=separation)
    ftemp['frame'] = i + start # Frame numbering
    return ftemp


# ── 3. Worker (knows about shared memory, calls track_frame) ────────────────

def _worker(i: int, shm_name: str, args: tuple, shape: tuple, dtype: np.dtype):
    """
    Attaches to shared memory and calls track_frame. 
    Do not call directly.
    """
    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    frames = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    frame = frames[i] # zero-copy view
    result = track_frame(i, frame, args)
    # detach (do not unlink — main owns it)
    shm.close()                           
    return i, result


# ── 4. Orchestration ─────────────────────────────────────────────────────────

def process_video(video_path: str, start: int, stop: int, args: tuple, n_workers: int = None):
    shm, shape, dtype = load_video_to_shared_memory(video_path, start, stop)
    n_frames = shape[0]

    worker = partial(_worker, shm_name=shm.name, args=args, shape=shape, dtype=dtype)

    # Get the initial time before tracking
    t0 = time.time()
    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker, range(n_frames))
    # except:
    #     print('An error happened')
    finally:
        release_shared_memory(shm)  # always clean up, even if tracking crashes

    results.sort(key=lambda x: x[0])     # map() preserves order, but sort is cheap insurance

    # Display the timing
    t1 = time.time()
    tt = t1 - t0
    print(f"Wall time elapsed for the actual tracking = {tt // 60} min {tt % 60} s")

    res = [r for _, r in results]
    res = pd.concat(res, ignore_index=True)

    return res


# # ===========================================================================================
# # Input required, given as an example
# # ===========================================================================================
# path2vid = '/home/nfares/postdoc/experiments/data/2026/20260427_day1_diffusion_SiO2_in_capillary_100um_20fps/diffusion_px_65nm_500fps.mp4'
# path2folder = '/home/nfares/postdoc/experiments/data/2026/20260427_day1_diffusion_SiO2_in_capillary_100um_20fps/'
# # path2folder = path2vid[:path2vid.find('fps')+len('fps')+1]

# vidname = path2vid[len(path2folder):path2vid.find('.mp4')]
# savename = path2folder + 'trajectories_' + vidname
# # print(savename)

# px = 65e-9
# fps = 500

# start = 0
# stop = 100 # len(frames)

# R1 = 15 # radius for spot tracking
# minmass = 3000
# separation = 15
# invert = False

# args = (R1, minmass, separation, invert)

# n_workers = 10


# # ===========================================================================================
# # Go
# # ===========================================================================================

# if __name__ == '__main__':
#     # Display number of CPU
#     print('Total number of CPUs = {}'.format(os.cpu_count()))
#     # Run
#     res = process_video(video_path=path2vid, start=start, stop=stop, args=args, n_workers=n_workers)
#     # Put in SI
#     # res['xpx'] = res['x']
#     # res['ypx'] = res['y']
#     res['x'] = res['x'] * px
#     res['y'] = res['y'] * px
#     print(res.head())
#     print(res['x'].iloc[0])