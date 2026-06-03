# -*- coding: utf-8 -*-

"""
Wrapper: utilities for frame modification, video saving and conversion, etc. 

Author: nicofares

Creation: 2026-05-29
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import remove_background

def plot_one_trajectory(idp, tracks, fps=60, px=65e-9, figsize=(18,4.5), save=False, savepath='./'):
    x = tracks[tracks['particle'] == idp]['x'].to_numpy() * px
    y = tracks[tracks['particle'] == idp]['y'].to_numpy() * px
    r = np.sqrt((x-np.nanmean(x)) ** 2 + (y-np.nanmean(y)) ** 2)
    theta = tracks[tracks['particle'] == idp]['theta_unwrapped'].to_numpy()
    t = np.arange(len(x)) / fps
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].plot(t, (x - np.nanmean(x)) * 1e6, ls='-', lw=1.5, color='tab:blue', label=r'$x$')
    axes[0].plot(t, (y - np.nanmean(y)) * 1e6, ls='-', lw=1.5, color='tab:red', label=r'$y$')
    axes[1].plot(t, r * 1e6, ls='-', lw=1.5, color='tab:blue')
    axes[2].plot(t, theta, ls='-', lw=1.5, color='tab:blue')
    axes[0].legend()
    axes[0].set(xlabel=r'Time [s]', ylabel=r'$x,y$ [\textmu m]')
    axes[1].set(xlabel=r'Time [s]', ylabel=r'$\sqrt{(x-x(0))^2 + (y - y(0))^2}$ [\textmu m]')
    axes[2].set(xlabel=r'Time [s]', ylabel=r'$\theta$ [°]')
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(savepath)


def check_one_image(i, frames, bg, save_props, save_labeled, display=True, save_result=False, savename='./temp/', format='png'):
    im2 = remove_background(frames[i], bg)
    props = save_props[i]
    labeled = save_labeled[i]

    df = f[f['frame'] == i]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im2, cmap="gray")

    # regionprops: centroid + orientation line
    for _, row in df.iterrows():
        cx, cy = row["x"], row["y"]
        theta_rad = np.radians(row["theta"])
        ev = np.array([np.cos(theta_rad), np.sin(theta_rad)])
        ax.plot(
            [cx - ARROW_LEN * ev[0], cx + ARROW_LEN * ev[0]],
            [cy - ARROW_LEN * ev[1], cy + ARROW_LEN * ev[1]],
            "r-", linewidth=1.5, 
        )
        ax.plot(cx, cy, "+", mec='tab:red', markersize=12, markeredgewidth=2)

    for p in props:
        # Contour of this particle
        particle_mask = labeled == p.label
        contours = find_contours(particle_mask, level=0.5)
        for cnt in contours:
            ax.plot(cnt[:, 1], cnt[:, 0], 'c-', linewidth=2)  # note: cnt is (row, col) → swap for plot

    ax.axis("off")
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.close()

    if save_result:
        savename = savename + str(i) + '.' + format
        fig.savefig(savename)