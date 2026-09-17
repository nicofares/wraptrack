"""
Inner-part localization
------------------------
The inner blob is a rotating cube: its shape and intensity are NOT constant
frame to frame, so we don't assume any fixed shape (no Gaussian fit here,
unlike the ring). Instead we use a simple, shape-agnostic threshold
segmentation, anchored to quantities we already trust from the ring fit:

1. Estimate the background level (mean, std) from pixels *outside* the ring
   (i.e. far enough from the center that we're clearly off-particle).
2. Threshold = background_mean + offset. Any pixel above this is "signal".
3. Restrict the search to a disk around the ring center, out to the radial
   valley between the inner blob and the ring (so we never pick up ring
   pixels as part of the inner object).
4. Keep only the largest connected component in that disk (robust to a few
   stray noise pixels sneaking above threshold).
5. From that mask: area (pixel count), intensity (integrated and mean, both
   background-subtracted), and position (intensity-weighted center of mass).

Depends on ring_tracking.py for the ring center/radius/radial profile.
"""

import numpy as np
from scipy.ndimage import label

from ring_tracking import locate_ring


def estimate_background(img, xc, yc, r_ring, margin_inner=15, margin_outer=45,
                         n_sigma_clip=3.0, n_iter=4, min_pixels=50,
                         edge_warn_frac=0.4, max_outer_expand=200):
    """
    Robust, LOCAL background estimate for one particle.

    Two changes vs. a naive "everything outside the ring" approach, both
    needed once a frame can contain several particles:

    1. Bounded annulus (r_ring+margin_inner < r <= r_ring+margin_outer)
       instead of "everything with r > r_ring". Using the *whole* rest of
       the image as background is fine for a single isolated particle, but
       in a multi-particle frame it can reach all the way into a
       neighboring particle (or its ring/halo) and bias the estimate.
       Restricting to a local annulus close to this particle keeps the
       estimate local, and neighbors are only a problem if they physically
       overlap this annulus.

    2. Iterative sigma-clipping + median/MAD (robust statistics) instead of
       a plain mean/std. Even a local annulus can catch part of a nearby
       particle's edge, a hot pixel, etc. Sigma-clipping throws out pixels
       that are outliers relative to the bulk of the annulus (repeatedly,
       since a first pass can be skewed by a lot of contaminated pixels),
       and the median/MAD of what remains is far less sensitive to any
       leftover outliers than a mean/std would be.

    Edge handling
    -------------
    Pixels outside the image can never be included by construction --
    `r` is only ever evaluated at actual pixel locations (np.indices over
    the image's own shape), so there's no risk of reading beyond the frame.
    What CAN happen if the particle sits near an edge is that the annulus
    gets truncated: part of the ring of pixels around it simply doesn't
    exist in the frame. That means fewer pixels, and asymmetric coverage
    (e.g. only the interior-facing side of the annulus is sampled), which
    can bias the estimate even though nothing crashes. We guard against
    this two ways:
      - `edge_effect` flag: True if the number of valid annulus pixels is
        below `edge_warn_frac` of what a full, unclipped annulus at this
        radius would contain -- i.e. the particle is close enough to the
        edge that the background estimate is based on a small, one-sided
        sample and may be less reliable.
      - If there aren't even `min_pixels` valid pixels (e.g. particle is
        right in a corner), the outer margin is progressively expanded
        (up to `max_outer_expand`) until enough pixels are found. If it's
        still not enough, we give up and return NaNs rather than silently
        computing a statistic from a handful of pixels.

    Returns
    -------
    bg_mean, bg_std, edge_effect : robust background level, spread
        estimate, and a bool flagging possible edge truncation.
    """
    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))
    r = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)

    outer = margin_outer
    annulus = (r > r_ring + margin_inner) & (r <= r_ring + outer)
    vals = img[annulus].astype(np.float64)

    # coverage check: how much of a full (unclipped) annulus at this
    # position actually exists inside the frame?
    theoretical_area = np.pi * ((r_ring + margin_outer) ** 2 - (r_ring + margin_inner) ** 2)
    coverage = annulus.sum() / theoretical_area if theoretical_area > 0 else 1.0
    edge_effect = coverage < edge_warn_frac

    # if truncation left too few pixels to trust at all, progressively
    # widen the outer margin (still bounded, so we don't fall back to
    # "the whole rest of the image" and reintroduce the multi-particle
    # contamination problem)
    while vals.size < min_pixels and outer < max_outer_expand:
        outer += 20
        annulus = (r > r_ring + margin_inner) & (r <= r_ring + outer)
        vals = img[annulus].astype(np.float64)

    if vals.size < min_pixels:
        return np.nan, np.nan, True

    # iterative sigma-clipping: repeatedly drop pixels far from the mean,
    # so a contaminating neighbor (or its bright halo) gets excluded even
    # if it dominates the annulus at first
    for _ in range(n_iter):
        if vals.size < min_pixels:
            break
        mean, std = vals.mean(), vals.std()
        if std == 0:
            break
        keep = np.abs(vals - mean) < n_sigma_clip * std
        if keep.sum() == vals.size:
            break
        vals = vals[keep]

    bg_mean = np.median(vals)
    bg_std = 1.4826 * np.median(np.abs(vals - bg_mean))  # MAD -> std-equivalent
    return bg_mean, bg_std, edge_effect


def find_boundary_radius(r_profile, profile, r_ring):
    """
    Radius of the valley between the inner-blob peak (near r=0) and the
    ring peak (at r_ring). Used as the outer edge of the inner-part search
    disk, so ring pixels are never mistaken for inner-part pixels.

    Returns None if no usable interior region exists (e.g. r_ring itself
    is degenerate/too small, or the profile is empty) -- callers should
    treat that as "can't process this frame" rather than crash.
    """
    if r_profile.size == 0 or not np.isfinite(r_ring):
        return None
    inner = r_profile < r_ring
    if not np.any(inner):
        return None
    idx_valley = np.argmin(profile[inner])
    return r_profile[inner][idx_valley]


def segment_inner_part(img, xc, yc, r_boundary, bg_mean, offset):
    """
    Threshold-segment the inner part inside the search disk, keeping only
    the largest connected component.

    Two-sided threshold: |pixel - bg_mean| > offset, not just
    (pixel - bg_mean) > offset. This matters for DIC images specifically --
    DIC contrast comes from the *gradient* of optical path difference along
    the shear direction, so a 3D object typically shows one side brighter
    than background and the OPPOSITE side darker than background (the
    classic shadow-cast look), with flat regions sitting at background
    regardless of their actual structure. A one-sided threshold would only
    ever pick up the bright lobe and silently miss the dark one -- biasing
    area, intensity, and (worst) center of mass, since that bias rotates
    along with the cube instead of staying fixed. Thresholding on the
    magnitude of the deviation from background catches both lobes as a
    single object when they're part of the same particle.
    """
    threshold = offset  # offset here is a magnitude, not added to bg_mean

    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))
    r = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)
    search_region = r <= r_boundary

    deviation = img.astype(np.float64) - bg_mean
    mask = search_region & (np.abs(deviation) > threshold)
    if not np.any(mask):
        return None, threshold

    labeled, n_labels = label(mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background label
    largest_label = sizes.argmax()
    obj_mask = labeled == largest_label

    return obj_mask, threshold


def inner_part_properties(img, obj_mask, bg_mean):
    """
    Area, intensity, and center of mass of the mask.

    Weighted by |pixel - bg_mean| (magnitude of deviation from background),
    not signed deviation -- a bright lobe and a dark lobe of the same
    particle both represent real signal, and weighting by signed value
    would have them partially cancel in the center-of-mass and integrated
    intensity, which is not what we want when both are part of one object.
    """
    ys, xs = np.nonzero(obj_mask)
    raw_vals = img[obj_mask].astype(np.float64)
    weights = np.abs(raw_vals - bg_mean)

    area = int(obj_mask.sum())
    total_weight = weights.sum()
    if total_weight > 0:
        com_x = np.sum(xs * weights) / total_weight
        com_y = np.sum(ys * weights) / total_weight
    else:
        com_x, com_y = xs.mean(), ys.mean()

    return {
        "area": area,
        "center_x": com_x,
        "center_y": com_y,
        "intensity_integrated": total_weight,
        "intensity_mean": raw_vals.mean(),
    }


def locate_inner_part(img, ring_result=None, n_std=3.0, offset=None,
                       bg_margin_inner=15, bg_margin_outer=45):
    """
    Full pipeline for one frame.

    ring_result : output of ring_tracking.locate_ring(img). Computed
                  internally if not provided (slightly slower if you're
                  calling both per-frame; better to compute once and pass
                  in to both this and the ring step).
    n_std       : threshold offset expressed as (this many) x background std,
                  used only if `offset` is not given explicitly.
    offset      : absolute intensity offset above background mean. If given,
                  overrides n_std.
    bg_margin_inner, bg_margin_outer : define the annulus (beyond the ring)
                  used for the local background estimate. Keep
                  bg_margin_outer well below the typical distance to the
                  nearest neighboring particle in a multi-particle frame.
    """
    if ring_result is None:
        ring_result = locate_ring(img)

    if not ring_result.get("valid", True):
        # ring fit itself was degenerate for this frame -- nothing
        # trustworthy to build the inner-part search on
        return None

    xc, yc, r_ring = ring_result["center_x"], ring_result["center_y"], ring_result["radius"]

    bg_mean, bg_std, edge_effect = estimate_background(
        img, xc, yc, r_ring, margin_inner=bg_margin_inner, margin_outer=bg_margin_outer
    )
    if np.isnan(bg_mean):
        # particle too close to the edge / too little unobstructed
        # background even after expanding the annulus -- can't give a
        # trustworthy result for this frame
        return None
    if offset is None:
        offset = n_std * bg_std

    r_boundary = find_boundary_radius(ring_result["r_profile"], ring_result["profile"], r_ring)
    if r_boundary is None:
        return None

    obj_mask, threshold = segment_inner_part(img, xc, yc, r_boundary, bg_mean, offset)
    if obj_mask is None:
        return None

    props = inner_part_properties(img, obj_mask, bg_mean)
    props.update({
        "threshold": threshold,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "r_boundary": r_boundary,
        "mask": obj_mask,
        "edge_effect": edge_effect,
    })
    return props


if __name__ == "__main__":
    from PIL import Image

    img = np.array(Image.open("/mnt/user-data/uploads/testim.png").convert("L"))
    ring_result = locate_ring(img)
    inner_result = locate_inner_part(img, ring_result)

    print("Inner part result:")
    for k, v in inner_result.items():
        if k == "mask":
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
