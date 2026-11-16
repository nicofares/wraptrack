"""
Plot one image with its tracking result.

Works directly on the `results` DataFrame from the notebook loop (after
link_trajectories), in pixels or after conversion to SI units.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_frame_tracking(img, results, frame, px=1.0, ysize=None, ax=None,
                        plot_trajectory=True, trail_length=None, show_inner=True,
                        show_ids=True, zoom=None, cmap="gray"):
    """
    Draw `img` with the ring fits, centers, inner-part COMs and particle
    IDs of `frame`, plus (optionally) each particle's trajectory so far.

    img      : 2D array, the image of that frame (e.g. stack[frame]).
    results  : DataFrame with columns frame, xs, ys, as, xc, yc and
               (after linking) particle.
    frame    : frame number, as in results["frame"].
    px       : size of one pixel in the DataFrame's units. 1.0 if results
               are still in pixels; your `px` if you already converted
               xs, ys, as, xc, yc to SI units.
    ysize    : image height in px used for the y flip (ys = ysize - y).
               Default img.shape[0], i.e. the same as in the loop.
    ax       : matplotlib Axes to draw on; a new figure if None.
    plot_trajectory : if True (default), also draw each particle's
               ring-center path up to this frame. False = only the
               particles of this frame.
    trail_length : with plot_trajectory, only the last N frames of the
               path (None = all).
    show_inner : draw the inner-part center of mass.
    show_ids : write the trackpy particle ID next to each particle.
    zoom     : optional margin in px: crop the view to the particles in
               this frame plus this margin. None = whole image.

    Returns the Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    if ysize is None:
        ysize = img.shape[0]

    def to_img(x, y):
        # DataFrame -> image pixel coordinates (undo unit conversion and y flip)
        return np.asarray(x) / px, ysize - np.asarray(y) / px

    ax.imshow(img, cmap=cmap)
    cur = results[results["frame"] == frame]
    has_id = "particle" in results.columns
    colors = plt.get_cmap("tab10")
    theta = np.linspace(0, 2 * np.pi, 200)

    for _, row in cur.iterrows():
        pid = int(row["particle"]) if has_id else None
        c = colors(pid % 10) if has_id else "tab:orange"
        if np.isnan(row["xs"]):
            continue
        cx, cy = to_img(row["xs"], row["ys"])
        a = row["as"] / px

        # ring fit and its center
        ax.plot(cx + a * np.cos(theta), cy + a * np.sin(theta), "-", c=c, lw=1.5)
        ax.plot(cx, cy, "+", mec=c, ms=12, mew=2)

        # inner-part center of mass
        if show_inner and not np.isnan(row.get("xc", np.nan)):
            ix, iy = to_img(row["xc"], row["yc"])
            ax.plot(ix, iy, "x", mec="lime", ms=8, mew=1.8)

        if show_ids and has_id:
            ax.text(cx + 0.75 * a, cy - 0.75 * a, str(pid), color=c, fontsize=11,
                    fontweight="bold")

        # path of this particle up to the current frame
        if plot_trajectory and has_id:
            path = results[(results["particle"] == pid) & (results["frame"] <= frame)]
            if trail_length is not None:
                path = path[path["frame"] > frame - trail_length]
            path = path.sort_values("frame")
            tx, ty = to_img(path["xs"], path["ys"])
            ax.plot(tx, ty, "-", c=c, lw=1, alpha=0.8)

    if zoom is not None and len(cur):
        xs, ys = to_img(cur["xs"], cur["ys"])
        r = (cur["as"] / px).max()
        ax.set_xlim(np.nanmin(xs) - r - zoom, np.nanmax(xs) + r + zoom)
        ax.set_ylim(np.nanmax(ys) + r + zoom, np.nanmin(ys) - r - zoom)

    ax.set_title(f"frame {frame}: {len(cur)} particle(s)")
    ax.axis("off")
    return ax


if __name__ == "__main__":
    # Example with the 10-frame test sequence
    import pandas as pd
    from PIL import Image

    results = pd.read_csv("trajectories_im0-9.csv")       # in pixels -> px=1
    frame = 5
    img = np.array(Image.open(f"im_{frame}.png").convert("L"))
    plot_frame_tracking(img, results, frame, px=1.0, zoom=40, plot_trajectory=False)
    plt.tight_layout()
    plt.show()
