"""
Multi-particle tracking
-----------------------
Extends the single-particle ring + inner-part pipeline to frames that
contain several particles, and links them into trajectories with trackpy.

Per frame:
  1. DETECT  -- rough candidate positions for every particle
                (detect_particles). Only needs to be within ~half a ring
                radius of the truth; precision comes from step 2.
  2. FIT     -- run locate_ring on a crop around each candidate, with
                RANSAC seeded at that candidate and only allowed to return
                circles centered near it (so a neighbor's ring in the crop
                can't be picked instead).
  3. CLEAN   -- drop invalid fits, fits that wandered too far from their
                seed, and duplicates (two seeds converging on the same
                ring -- happens with elongated or two-lobed blobs).
  4. INNER   -- run locate_inner_part on a crop around each ring, with the
                other particles' disks excluded from the background
                estimate.
Then, over the whole stack:
  5. LINK    -- trackpy.link on the ring centers -> "particle" column.

The single-particle modules are reused as-is apart from small, backward-
compatible additions (seed_center / max_center_offset / ransac_kwargs in
ring_tracking, exclude_mask in inner_part_tracking). Calling them exactly
as before still gives the same results.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import fftconvolve

from ring_tracking import locate_ring, crop_around_point
from inner_part_tracking import locate_inner_part


# ----------------------------------------------------------------------
# 1. Detection
# ----------------------------------------------------------------------
def _hough_ring_accumulator(img, ring_radius, scales, smoothing, grad_k, acc_sigma, pad, bg_sigma):
    """
    Gradient-directed circular Hough transform. Every significant edge
    pixel (gradient magnitude > median + grad_k * MAD) casts one vote at
    distance s * ring_radius along its gradient, in both directions, for
    each scale s. Returns the blurred accumulator on a canvas padded by
    `pad` px on every side, normalized so that 1.0 = the votes a complete,
    sharp ring of radius ring_radius would cast at its center at ONE scale.
    """
    img = img.astype(np.float64)
    sm = gaussian_filter(img - gaussian_filter(img, bg_sigma, mode="nearest"), smoothing)
    gy, gx = np.gradient(sm)
    gm = np.hypot(gx, gy)
    med = np.median(gm)
    mad = 1.4826 * np.median(np.abs(gm - med))
    ys, xs = np.nonzero(gm > med + grad_k * mad)
    ux, uy = gx[ys, xs] / gm[ys, xs], gy[ys, xs] / gm[ys, xs]
    Ny, Nx = img.shape
    H, W = Ny + 2 * pad, Nx + 2 * pad
    acc = np.zeros(H * W)
    for sc in scales:
        for sign in (1, -1):
            cx = np.rint(xs + sign * sc * ring_radius * ux).astype(int) + pad
            cy = np.rint(ys + sign * sc * ring_radius * uy).astype(int) + pad
            ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            acc += np.bincount(cy[ok] * W + cx[ok], minlength=H * W)
    acc = gaussian_filter(acc.reshape(H, W), acc_sigma)
    # a full ring has two edges (inner + outer) of length ~2 pi R each, all
    # voting into a blob of width acc_sigma
    full = 4 * np.pi * ring_radius / (2 * np.pi * acc_sigma ** 2)
    return acc / full


def detect_particles(img, ring_radius, method="hough", blur_sigma=None, min_separation=None,
                     threshold_k=8.0, max_particles=None, use_abs=True,
                     exclude_border=0, bg_sigma=None, ring_width=None, disk_frac=0.6,
                     ring_weight=0.5, radius_scales=None, outside=0.0, min_template_coverage=0.4,
                     min_votes=0.65, min_votes_outside=0.4, grad_k=4.0, return_scores=False):
    """
    Rough candidate positions for all particles in a frame.

    method : "hough" (default), "template" or "blob".
        "hough"    -- circular Hough transform on edge pixels (see
                      _hough_ring_accumulator): each significant edge pixel
                      votes for a center one ring radius away along its
                      gradient. A ring's edges all vote for its center; a
                      blob's edge (smaller radius b) votes onto circles of
                      radius R - b and R + b, i.e. no peak. Each edge pixel
                      votes ONCE regardless of brightness, so a particle
                      with a dim inner part is found next to one with a
                      bright inner part -- the "template" and "blob"
                      methods respond to brightness and missed exactly
                      that case on a real 6-particle chain (2 of 6 were not
                      even local maxima). Bare ringless blobs give no peak.
                      Votes may land outside the image, so particles cut by
                      the border are found too (see `outside`).
                      Thresholds are in units of "a complete ring's votes"
                      (scores above 1 are normal: several radius scales
                      vote at once). min_votes (0.65) for centers inside
                      the image -- measured on real data: weakest real ring
                      0.78, strongest non-ring 0.54 (synthetic) / 0.95 (a
                      bare blob, which the fit then rejects). For centers
                      OUTSIDE the image min_votes_outside (0.4): those are
                      only ever masked, never reported.
        "blob"     -- blur and keep local maxima. Finds bright patches; but
                      the contact point of two touching rings is ALSO a
                      bright patch (the two rings add up there), and it can
                      outshine both particles' blobs -- seen on a synthetic
                      tangent pair: one candidate at the midpoint instead
                      of two at the centers.
        "template" -- correlate with what a particle looks like: a central
                      disk (radius disk_frac * ring_radius, the inner blob)
                      PLUS an annulus at ring_radius (the ring), each
                      normalized, mixed with weight ring_weight for the
                      ring. At a true center both parts match fully; at a
                      contact point the disk only sees two arc segments
                      and the annulus mostly misses, so the response is
                      much weaker. Also unbiased by asymmetric DIC lobes
                      (the ring term is centered on the ring), so
                      candidates land within ~1 px instead of ~6 px.
                      The template is applied at several radii
                      (radius_scales x ring_radius) and the best response
                      per pixel is kept, so particles of different sizes
                      in one frame are all matched (seen on real data: a
                      23.4 px ring missed by a single 27 px template).
                      Scales only go DOWN from ring_radius: a template
                      LARGER than the real rings responds strongly at the
                      contact point of two touching rings again (seen on
                      a synthetic tangent pair with a 1.15x scale). So set
                      ring_radius to the LARGEST ring radius you expect.

    use_abs : work on |img - background| (default), so bright and dark DIC
        lobes both count as signal.
    bg_sigma : scale of the smooth background subtracted first, default
        4 * ring_radius (removes illumination gradients across the field).
    blur_sigma : "blob" method only, default 0.3 * ring_radius.
    ring_width : "template" method, Gaussian width of the annulus, default
        max(2, 0.12 * ring_radius).
    min_separation : minimum distance between two candidates, px, circular
        neighborhood. Default 0.8 * ring_radius: two touching particles of
        radius ~R are ~2R apart, and each must NOT fall inside the other's
        suppression circle even if ring_radius is overestimated a bit.
    threshold_k : a peak is kept if it exceeds median + threshold_k * MAD
        of the filtered image. Raise if you get spurious detections, lower
        if faint particles are missed.
    max_particles : optional cap; keeps the strongest peaks.
    outside : "template" method only. Also look for particles whose CENTER
        lies up to this many px outside the image (cut by the border).
        The response is normalized by the fraction of the template that
        falls inside the image, so a half-visible particle scores like a
        whole one. Default 0 (centers inside the image only). Peaks where
        less than min_template_coverage of the template is inside the
        image are ignored (too little evidence).
    exclude_border : drop candidates closer than this to the image edge.

    Returns an (N, 2) array of (x, y) in image coordinates, strongest first.
    """
    img = img.astype(np.float64)
    if bg_sigma is None:
        bg_sigma = 4 * ring_radius
    sig = img - gaussian_filter(img, bg_sigma, mode="nearest")
    if use_abs:
        sig = np.abs(sig)
    if min_separation is None:
        min_separation = 0.8 * ring_radius

    if method == "hough":
        if radius_scales is None:
            radius_scales = (0.85, 1.0, 1.15)
        pad = int(np.ceil(outside))
        acc = _hough_ring_accumulator(
            img, ring_radius, radius_scales, smoothing=float(np.clip(0.09 * ring_radius, 1.0, 1.5)),
            grad_k=grad_k, acc_sigma=max(1.0, 0.1 * ring_radius), pad=pad, bg_sigma=bg_sigma)
        m = int(np.ceil(min_separation))
        yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
        footprint = xx ** 2 + yy ** 2 <= min_separation ** 2
        is_peak = acc == maximum_filter(acc, footprint=footprint)
        py, px = np.nonzero(is_peak & (acc > min(min_votes, min_votes_outside)))
        Ny, Nx = img.shape
        x, y = px - pad, py - pad
        inside = (x >= 0) & (x <= Nx - 1) & (y >= 0) & (y <= Ny - 1)
        keep = np.where(inside, acc[py, px] > min_votes, acc[py, px] > min_votes_outside)
        if exclude_border > 0:
            b = exclude_border
            keep &= ~inside | ((x >= b) & (x <= Nx - 1 - b) & (y >= b) & (y <= Ny - 1 - b))
        x, y, sc = x[keep], y[keep], acc[py, px][keep]
        order = np.argsort(sc)[::-1]
        out = np.column_stack([x[order], y[order]]).astype(np.float64)
        if max_particles is not None:
            out, sc = out[:max_particles], sc[order][:max_particles]
        else:
            sc = sc[order]
        return (out, sc) if return_scores else out

    if radius_scales is None:
        radius_scales = (1.0,)
    if method == "blob":
        if blur_sigma is None:
            blur_sigma = 0.3 * ring_radius
        filtered = gaussian_filter(sig, blur_sigma)
    elif method == "template":
        pad = int(np.ceil(outside))
        if pad > 0:
            # zero = background, since the smooth background was subtracted
            sig = np.pad(sig, pad, mode="constant")
        inside = np.zeros(sig.shape)
        inside[pad:sig.shape[0] - pad, pad:sig.shape[1] - pad] = 1.0
        filtered = None
        coverage = None
        for scale in radius_scales:
            R = scale * ring_radius
            w = ring_width if ring_width is not None else max(2.0, 0.12 * R)
            n = int(np.ceil(R + 3 * w))
            yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
            rr = np.hypot(xx, yy)
            ring = np.exp(-(rr - R) ** 2 / (2 * w ** 2))
            ring /= ring.sum()
            disk = (rr <= disk_frac * R).astype(np.float64)
            disk /= disk.sum()
            kernel = ring_weight * ring + (1 - ring_weight) * disk
            cov = fftconvolve(inside, kernel, mode="same")
            resp = fftconvolve(sig, kernel, mode="same") / np.maximum(cov, 1e-6)
            resp[cov < min_template_coverage] = 0.0
            filtered = resp if filtered is None else np.maximum(filtered, resp)
            coverage = cov if coverage is None else np.maximum(coverage, cov)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'template' or 'blob'")

    if method == "template" and pad > 0:
        core = filtered[pad:filtered.shape[0] - pad, pad:filtered.shape[1] - pad]
    else:
        pad, core = 0, filtered
    med = np.median(core)
    mad = 1.4826 * np.median(np.abs(core - med))
    thresh = med + threshold_k * mad

    m = int(np.ceil(min_separation))
    yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
    footprint = xx ** 2 + yy ** 2 <= min_separation ** 2
    is_peak = (filtered == maximum_filter(filtered, footprint=footprint)) & (filtered > thresh)

    if exclude_border > 0:
        b = int(exclude_border)
        is_peak[:b, :] = is_peak[-b:, :] = False
        is_peak[:, :b] = is_peak[:, -b:] = False

    ys, xs = np.nonzero(is_peak)
    order = np.argsort(filtered[ys, xs])[::-1]
    xs, ys = xs[order] - pad, ys[order] - pad
    if max_particles is not None:
        xs, ys = xs[:max_particles], ys[:max_particles]
    return np.column_stack([xs, ys]).astype(np.float64)


# ----------------------------------------------------------------------
# 2-4. Fit all particles in one frame
# ----------------------------------------------------------------------
def _disk_mask(shape, xc, yc, radius):
    y_idx, x_idx = np.indices(shape)
    return (x_idx - xc) ** 2 + (y_idx - yc) ** 2 <= radius ** 2


def _pixel_noise(img, region):
    """
    Pixel noise level from differences of horizontally adjacent pixels
    that both lie in `region`: 1.4826 * MAD(diff) / sqrt(2).

    NOT the spread of the raw values: in a background region that also
    holds a slow illumination gradient or the edge of a particle's halo,
    the raw spread measures those, not the noise. On a nearly noise-free
    crop (pixel noise ~2) the raw spread over-estimated it several-fold,
    and the over-noisy fill then dominated the fits next to it.
    """
    pair = region[:, 1:] & region[:, :-1]
    d = (img[:, 1:] - img[:, :-1])[pair]
    if d.size < 20:
        vals = img[region]
        return 1.4826 * np.median(np.abs(vals - np.median(vals)))
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def isolate_particle(frame, own, others, reach, bg_window, seed=0):
    """
    Return a copy of `frame` in which the pixels belonging to NEIGHBORING
    particles are replaced by local background + matched noise, so the
    single-particle fits only ever see this particle.

    Why: each fit (RANSAC edges, 2D ring fit, radial profile, inner part)
    assumes one particle per crop. When two particles touch, the
    neighbor's ring and bright blob sit right inside this particle's fit
    annulus, and no crop size or center constraint fixes that -- seen on
    a real tangent pair: the dimmer particle's fit either slid onto the
    neighbor's ring or converged 19 px off with an inner area of 2 px.

    Which pixels belong to a neighbor: a power-diagram partition (Voronoi
    weighted by radius). Pixel p belongs to particle k minimizing
    |p - c_k|^2 - r_k^2. For two touching rings this boundary passes
    through their contact point, i.e. each ring keeps exactly its own
    pixels; with unequal radii it shifts toward the smaller particle, as
    it should. Pixels are only reassigned within (r_k + reach) of the
    neighbor, so open background stays untouched.

    Why background + NOISE, not just a constant: SNR and the inner part's
    threshold are built on background noise estimates; a flat patch would
    shrink those estimates and inflate SNR. Noise is drawn with a fixed
    seed, so the same frame always gives the same result.

    own    : (x, y, r) of the particle to isolate.
    others : list of (x, y, r) -- or (x, y, r, reach) to override `reach`
             for that neighbor -- of its neighbors.
    reach  : how far beyond a neighbor's ring radius its pixels extend
             (ring tail + halo).
    bg_window : background level/noise are estimated from pixels within
             this distance of `own`, outside every particle's (r + reach).
    """
    img = frame.astype(np.float64)
    if not others:
        return img
    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))

    def pdist(x, y, r):
        return (x_idx - x) ** 2 + (y_idx - y) ** 2 - r ** 2

    own_pd = pdist(*own)
    foreign = np.zeros((Ny, Nx), dtype=bool)
    occupied = (x_idx - own[0]) ** 2 + (y_idx - own[1]) ** 2 <= (own[2] + reach) ** 2
    for o in others:
        x, y, r = o[:3]
        reach_o = o[3] if len(o) > 3 else reach   # optional per-neighbor reach
        near = (x_idx - x) ** 2 + (y_idx - y) ** 2 <= (r + reach_o) ** 2
        foreign |= near & (pdist(x, y, r) < own_pd)
        occupied |= near
    if not foreign.any():
        return img

    local = ((x_idx - own[0]) ** 2 + (y_idx - own[1]) ** 2 <= bg_window ** 2) & ~occupied
    if local.sum() < 50:
        local = ~occupied
    bg = np.median(img[local])
    noise = _pixel_noise(img, local)

    rng = np.random.default_rng(seed)
    img[foreign] = bg + noise * rng.standard_normal(foreign.sum())
    return img


def ring_coverage(img, xc, yc, r0, sigma, A, B, n_sectors=16, frac=0.3):
    """
    Fraction of angular sectors (n_sectors around the ring) whose mean
    intensity in the ring band |r - r0| < max(1, sigma) exceeds
    B + frac * A, where A, B are the fitted ring amplitude and baseline.

    A real ring is bright all the way round (measured 0.88-1.0 on real
    particles, touching ones included once neighbors are masked). A
    "ring" fitted through some other bright object -- e.g. a circle
    passing through a bare, ringless blob -- is bright over one short arc
    only (measured 0.12). The 2D fit itself can't tell the two apart:
    the model only cares about the radial profile on average.
    """
    y, x = np.indices(img.shape)
    r = np.hypot(x - xc, y - yc)
    th = np.arctan2(y - yc, x - xc)
    band = np.abs(r - r0) < max(1.0, sigma)
    sec = ((th + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    vals = [img[band & (sec == k)].mean() for k in range(n_sectors) if np.any(band & (sec == k))]
    if not vals:
        return 0.0
    return float(np.mean(np.array(vals) > B + frac * A))


def mask_particles(frame, particles, reach, seed=0):
    """
    Copy of `frame` with every particle in `particles` (list of (x, y, r))
    replaced by background + matched noise out to r + reach. Used to look
    for particles that detection missed: whatever is still there after
    removing everything already found.
    """
    img = frame.astype(np.float64)
    if not particles:
        return img
    y_idx, x_idx = np.indices(img.shape)
    occupied = np.zeros(img.shape, dtype=bool)
    for (x, y, r) in particles:
        occupied |= (x_idx - x) ** 2 + (y_idx - y) ** 2 <= (r + reach) ** 2
    vals = img[~occupied]
    if vals.size < 50:
        return img
    bg = np.median(vals)
    noise = _pixel_noise(img, ~occupied)
    rng = np.random.default_rng(seed)
    img[occupied] = bg + noise * rng.standard_normal(occupied.sum())
    return img


def locate_particles(frame, ring_radius, ring_kwargs=None, inner_kwargs=None,
                     detect_kwargs=None, candidates=None, crop_half_size=None,
                     max_shift=None, dedup_dist=None, exclusion_margin=None,
                     radius_range=(0.75, 1.3), isolate=True, reach=None, n_passes=3,
                     sigma_max_frac=0.15, max_drift=None, max_rounds=1, min_coverage=0.7,
                     ringless_frac=0.4):
    """
    Locate every particle (ring + inner part) in one frame.

    ring_radius : approximate ring radius in px. Sets the scale for
        detection, crop sizes and all the distance checks. Only needs to be
        roughly right (within ~20%); the ring radius itself is fitted.
    ring_kwargs : passed to locate_ring for every particle (min_snr,
        smoothing_sigma, range_lower_px, range_upper_px, fixed_params, ...).
    inner_kwargs : passed to locate_inner_part (n_std, offset,
        bg_margin_inner, bg_margin_outer).
    detect_kwargs : passed to detect_particles.
    candidates : optional (N, 2) array of approximate (x, y) positions. If
        given, detection is skipped.
    crop_half_size : half-size of the crop each ring fit runs on. Default
        2.7 * ring_radius.

    Scale: every distance default here is a multiple of ring_radius, so the
    same call works at any magnification (tested from R ~ 11 to ~ 29 px).
    The exceptions you may want to set yourself are noted in ring_kwargs /
    inner_kwargs defaults below: edge smoothing (default clip(0.09 R, 1,
    1.5) px), inner-part background margins (0.5 R and 1.6 R).
    max_shift : reject a fit whose center ends up farther than this from
        its detected position. Default 0.5 * ring_radius. Also passed to RANSAC as
        max_center_offset.
    dedup_dist : two accepted fits closer than this are the same particle;
        the higher-SNR one is kept. Default ring_radius.
    exclusion_margin : neighbors' disks of radius (r_ring + margin) are
        excluded from each particle's background estimate. Default
        bg_margin_inner from inner_kwargs (15).
    radius_range : (low, high) multiples of ring_radius. RANSAC only looks
        for circles in this radius range, and ring fits outside it are
        rejected (guards against fitting the bright blob's own edge as
        the ring -- seen on a chain particle: 18.9 px instead of 25.5 with
        the previous (0.6, 1.5)). None to disable.
    min_coverage : a fit is rejected unless at least this fraction of the
        ring's angular sectors is actually bright (see ring_coverage).
        Rejects circles fitted through ringless bright objects. Default
        0.7: real rings measured 0.75-1.0 (touching ones included), fits
        through a bare blob 0.12-0.62. None to disable.
    isolate : if True (default), neighbors' pixels are replaced by
        background before fitting each particle (see isolate_particle).
        Only has an effect when particles are close.
    reach : extent of a particle beyond its ring radius, for isolation.
        Default 0.5 * ring_radius.
    sigma_max_frac : the ring width is bounded to [1 px, sigma_max_frac *
        ring_radius] in the 2D fit, the radius to radius_range, and the
        center to a +/- max_shift box around its seed. These bounds are
        what keep a fully free fit (radius AND width free) from absorbing
        the bright inner blob's tail into a wide "ring". Measured widths
        are ~0.1 x radius, so 0.15 leaves margin. Not applied to
        parameters you fix via fixed_params.
    max_drift : a fit may end up at most this far from the DETECTED
        position of its candidate. Default max_shift (0.5 * ring_radius).
        The Hough detector puts candidates within ~2-5 px of the true
        center; a fit that wanders farther has almost always latched onto
        a circle spanning two neighbors (seen on a real chain: a radius-34
        "ring" 20 px off, wedged between two particles).
    max_rounds : "find, mask, find again". After fitting, every validated
        particle is masked out and detection is run again on what is left;
        new candidates are fitted (with neighbors isolated) and the loop
        repeats, up to max_rounds detection rounds in total. Recovers
        particles that detection misses next to a brighter neighbor.
        1 = single detection, as before.
    n_passes : total number of fitting passes when particles are close.
        Pass 1 fits every candidate on its own. Each further pass refits
        every candidate with the particles VALIDATED in the previous pass
        masked out, using their fitted centers and radii. Default 3: pass 2
        recovers particles hidden by a neighbor, pass 3 refits everyone
        with the now-complete set of neighbors. Extra passes only run when
        some candidates are actually close.

    Returns a list of dicts, one per particle, all positions in pixel,
    full-image, image (row-down) coordinates. Each dict holds the ring
    result ("ring"), the inner-part result ("inner", or None), the crop
    offset of the inner mask ("mask_offset"), and "isolated" (True if
    neighbor pixels were masked for this particle).
    """
    ring_kwargs = dict(ring_kwargs or {})
    inner_kwargs = dict(inner_kwargs or {})
    detect_kwargs = dict(detect_kwargs or {})

    if crop_half_size is None:
        crop_half_size = int(round(2.7 * ring_radius))
    if max_shift is None:
        max_shift = 0.5 * ring_radius
    if dedup_dist is None:
        dedup_dist = ring_radius
    if reach is None:
        reach = 0.5 * ring_radius
    if max_drift is None:
        max_drift = max_shift
    # inner-part background annulus: was fixed at 15 / 45 px, which at small
    # magnification (R ~ 11 px) reaches past the whole image
    inner_kwargs.setdefault("bg_margin_inner", 0.5 * ring_radius)
    inner_kwargs.setdefault("bg_margin_outer", 1.6 * ring_radius)
    bg_margin_inner = inner_kwargs["bg_margin_inner"]
    bg_margin_outer = inner_kwargs["bg_margin_outer"]
    # edge smoothing for RANSAC: 1.5 px blurs a small, thin ring away
    ring_kwargs.setdefault("smoothing_sigma", float(np.clip(0.09 * ring_radius, 1.0, 1.5)))
    if exclusion_margin is None:
        exclusion_margin = bg_margin_inner

    fixed = ring_kwargs.get("fixed_params") or {}
    r_fixed = fixed.get("a", fixed.get("radius", fixed.get("r0")))
    if "ransac_kwargs" not in ring_kwargs:
        if r_fixed is not None:
            ring_kwargs["ransac_kwargs"] = {"min_radius": 0.7 * r_fixed, "max_radius": 1.3 * r_fixed}
        elif radius_range is not None:
            ring_kwargs["ransac_kwargs"] = {"min_radius": radius_range[0] * ring_radius,
                                            "max_radius": radius_range[1] * ring_radius}
    ring_kwargs.setdefault("max_center_offset", max_shift)
    # physical bounds for the free 2D fit + a start at the expected radius
    if radius_range is not None and r_fixed is None:
        ring_kwargs.setdefault("r0_bounds", (radius_range[0] * ring_radius, radius_range[1] * ring_radius))
        # several starting radii around the expected one: a single start
        # at ring_radius fails on a faint ring when the guess is >~10% off
        # (seen on real data, guess 25 vs true 29: fit lost; guess 31: a
        # wrong solution won). The multi-start keeps the best chi-square.
        ring_kwargs.setdefault("extra_r0_guesses", [f * ring_radius for f in (0.85, 1.0, 1.15)])
    if "sigma" not in fixed and sigma_max_frac is not None:
        # lower bound 0.5 px: small rings are thinner than 1 px (seen at
        # R ~ 11 px: fits pinned at a 1 px lower bound)
        ring_kwargs.setdefault("sigma_bounds", (0.5, max(1.5, sigma_max_frac * ring_radius)))
    ring_kwargs.setdefault("center_box", max_shift)

    # ---- 1. detection ----
    # Particles cut by the image border (center outside the frame) are
    # detected too, but only to be MASKED as neighbors -- never fitted or
    # reported. Without this, a half-visible particle touching a fully
    # visible one contaminates its fit (seen on a real chain crop: both
    # edge-adjacent particles lost).
    outside_cands = np.zeros((0, 2))
    if candidates is None:
        detect_kwargs.setdefault("outside", 0.9 * ring_radius)
        candidates = detect_particles(frame, ring_radius, **detect_kwargs)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    Ny, Nx = frame.shape
    in_frame = ((candidates[:, 0] >= 0) & (candidates[:, 0] <= Nx - 1) &
                (candidates[:, 1] >= 0) & (candidates[:, 1] <= Ny - 1))
    outside_cands = candidates[~in_frame]
    candidates = candidates[in_frame]
    r_default = r_fixed if r_fixed is not None else ring_radius
    interaction = 2 * (r_default + reach)  # closer than this -> they can see each other

    # candidates whose fit was rejected for poor azimuthal coverage: a
    # well-formed circle was found, but it isn't bright all the way round
    # -- the signature of a ringless object (e.g. a bare bright blob)
    ringless = set()

    def fit_one(i, seed_xy, neighbors):
        """Fit candidate i, seeded at seed_xy, with `neighbors` (list of
        (x, y, r) of other particles / candidates) masked out."""
        x0, y0 = candidates[i]
        own = (seed_xy[0], seed_xy[1], r_default)
        nb = [o for o in neighbors if np.hypot(o[0] - own[0], o[1] - own[1]) < interaction]
        img_i = (isolate_particle(frame, own, nb, reach, bg_window=crop_half_size + ring_radius, seed=i)
                 if nb else frame)
        res = locate_ring(img_i, approx_center=seed_xy, half_size=crop_half_size, **ring_kwargs)
        ok = (res is not None and res.get("valid", False)
              and np.isfinite(res["center_x"]) and np.isfinite(res["radius"]))
        if ok and radius_range is not None and r_fixed is None:
            ok = radius_range[0] * ring_radius <= res["radius"] <= radius_range[1] * ring_radius
        # wandered too far from its DETECTED position -> not this particle
        if ok:
            ok = np.hypot(res["center_x"] - x0, res["center_y"] - y0) <= max_drift
        if ok and min_coverage is not None:
            ok = ring_coverage(img_i, res["center_x"], res["center_y"], res["radius"], res["sigma"],
                               res["intensity_peak"], res["baseline"]) >= min_coverage
            if not ok:
                ringless.add(i)
            else:
                ringless.discard(i)
        return (res if ok else None), img_i, bool(nb)

    def validated(fits):
        """Indices of successful fits, duplicates removed (highest SNR wins)."""
        order = sorted([i for i in range(len(fits)) if fits[i][0] is not None],
                       key=lambda i: -np.nan_to_num(fits[i][0]["snr"], nan=-np.inf))
        kept = []
        for i in order:
            r = fits[i][0]
            if all(np.hypot(r["center_x"] - fits[k][0]["center_x"],
                            r["center_y"] - fits[k][0]["center_y"]) > dedup_dist for k in kept):
                kept.append(i)
        return kept

    def as_xyr(f):
        return (f[0]["center_x"], f[0]["center_y"], f[0]["radius"])

    def neighbor_estimates(best, n):
        """
        What to mask for each fit: validated particles at their FITTED
        center and radius, plus every candidate not validated (yet) at its
        DETECTED position with r = ring_radius -- except candidates sitting
        on a validated particle (duplicates, or a contact-point artifact
        whose fit converged onto a real ring).

        Using unvalidated candidates matters in dense clusters: in a chain
        every particle's crop is filled by its neighbors, so NO particle
        passes on its own, and masking only validated neighbors never gets
        started (seen on a real 3-particle chain: 0 found). Bare ringless
        blobs get masked too -- they are real objects, so that's welcome.
        """
        est = {i: as_xyr(f) for i, f in best.items()}
        validated_xyr = list(est.values())
        for j in range(n):
            if j in est:
                continue
            cx, cy = candidates[j]
            if all(np.hypot(cx - v[0], cy - v[1]) > 1.2 * r_default for v in validated_xyr):
                # a candidate known to be ringless (failed on coverage) is
                # masked tightly: treating a bare blob as a full-size
                # particle cut off the ring of the real particle touching
                # it (seen on a real chain). Any other unvalidated
                # candidate is masked as a full-size particle.
                if j in ringless:
                    est[j] = (cx, cy, ringless_frac * r_default, 0.15 * r_default)
                else:
                    est[j] = (cx, cy, r_default)
        # border-cut particles: always masked, never fitted
        for k, (cx, cy) in enumerate(outside_cands):
            est[("out", k)] = (cx, cy, r_default)
        return est

    # best fit so far per candidate index (only validated, deduplicated)
    best = {}
    n_done = 0
    for round_ in range(max_rounds):
        n = len(candidates)
        new_idx = list(range(n_done, n))
        if not new_idx:
            break

        # ---- 2. ring fits ----
        # Pass 1: each new candidate fitted with every OTHER candidate /
        # known particle masked out (see neighbor_estimates).
        known = neighbor_estimates(best, n) if isolate else {}
        fits = {i: fit_one(i, candidates[i], [v for k, v in known.items() if k != i]) for i in new_idx}
        fits.update(best)
        all_fits = [fits.get(i, (None, frame, False)) for i in range(n)]
        best = {i: all_fits[i] for i in validated(all_fits)}

        # Passes 2..n_passes: refit EVERY candidate (failed ones included --
        # a particle hidden by a touching neighbor typically fails pass 1)
        # with its neighbors masked out, now at their FITTED positions and
        # radii where available. A particle's best fit is kept across
        # passes, so failing ONE pass (e.g. while a neighbor's estimate is
        # still poor) doesn't make it vanish.
        allpos = np.vstack([candidates, outside_cands])
        close = any(np.hypot(*(allpos[i] - allpos[j])) < interaction
                    for i in range(n) for j in range(i + 1, len(allpos)))
        if isolate and close:
            for _ in range(n_passes - 1):
                known = neighbor_estimates(best, n)
                new_fits = []
                for i in range(n):
                    seed_xy = known[i][:2] if i in best else candidates[i]
                    nb = [v for k, v in known.items() if k != i]
                    new_fits.append(fit_one(i, seed_xy, nb))
                for i in validated(new_fits):
                    best[i] = new_fits[i]
                all_fits = [best.get(i, (None, frame, False)) for i in range(n)]
                best = {i: all_fits[i] for i in validated(all_fits)}

        n_done = n
        # ---- find again: mask everything found, detect in what's left ----
        if round_ == max_rounds - 1:
            break
        residual = mask_particles(frame, [as_xyr(f) for f in best.values()], reach, seed=round_)
        extra = detect_particles(residual, ring_radius, **detect_kwargs)
        # ignore anything sitting on a particle already found or tried
        taken = [as_xyr(f)[:2] for f in best.values()] + [tuple(c) for c in candidates]
        extra = [c for c in extra
                 if all(np.hypot(c[0] - t[0], c[1] - t[1]) > dedup_dist for t in taken)]
        if not extra:
            break
        candidates = np.vstack([candidates, np.array(extra)])

    kept = list(best.keys())
    fits = best

    # ---- 4. inner part per ring, on the SAME isolated image, neighbors excluded ----
    particles = []
    for i in kept:
        ring, img_i, was_isolated = fits[i]
        half = int(np.ceil(ring["radius"] + bg_margin_outer + 2))
        crop, (x_off, y_off) = crop_around_point(img_i, ring["center_x"], ring["center_y"], half)

        exclude = np.zeros(crop.shape, dtype=bool)
        for k in kept:
            if k == i:
                continue
            other = fits[k][0]
            exclude |= _disk_mask(crop.shape, other["center_x"] - x_off, other["center_y"] - y_off,
                                  other["radius"] + exclusion_margin)

        ring_local = dict(ring, center_x=ring["center_x"] - x_off, center_y=ring["center_y"] - y_off)
        inner = locate_inner_part(crop, ring_local, exclude_mask=exclude, **inner_kwargs)
        if inner is not None:
            inner = dict(inner, center_x=inner["center_x"] + x_off, center_y=inner["center_y"] + y_off)

        particles.append({"ring": ring, "inner": inner, "mask_offset": (x_off, y_off),
                          "isolated": was_isolated})

    return particles


def particles_to_rows(particles, frame_index, fps, ysize):
    """
    Convert locate_particles output to rows with the same columns as the
    single-particle notebook (pixels, y flipped to the "up" convention).
    A particle whose inner part failed keeps its ring values and gets NaN
    for the inner columns, instead of the whole row being lost.
    """
    rows = []
    for p in particles:
        ring, inner = p["ring"], p["inner"]
        rows.append({
            "frame": frame_index,
            "time": frame_index / fps,
            "xs": ring["center_x"],
            "ys": ysize - ring["center_y"],
            "as": ring["radius"],
            "sigmas": ring["sigma"],
            "Is_max": ring["intensity_peak"],
            "Is": ring["intensity_integrated"],
            "snr": ring["snr"],
            "xc": inner["center_x"] if inner else np.nan,
            "yc": ysize - inner["center_y"] if inner else np.nan,
            "Ac": inner["area"] if inner else np.nan,
            "Ic": inner["intensity_integrated"] if inner else np.nan,
            "edge_effect": inner["edge_effect"] if inner else np.nan,
        })
    return rows


# ----------------------------------------------------------------------
# 5. Linking
# ----------------------------------------------------------------------
def link_trajectories(df, search_range, memory=3, min_length=None, **link_kwargs):
    """
    Link per-frame detections into trajectories with trackpy.

    Links on the RING centers (xs, ys) -- the ring fit is the precise,
    stable one; the inner COM moves around as the cube rotates. Run this
    on pixel coordinates, BEFORE converting to SI units, so search_range
    is in px.

    search_range : max displacement between frames, px. Should be larger
        than the largest real step but smaller than half the typical
        distance between particles, otherwise identities can swap.
    memory : a particle may vanish for this many frames (failed fit,
        crossing) and still keep its ID when it reappears.
    min_length : optional; drop trajectories shorter than this (trackpy
        filter_stubs).

    Returns a copy of df with a "particle" column.
    """
    import trackpy as tp

    out = tp.link(df, search_range, memory=memory, pos_columns=["xs", "ys"],
                  t_column="frame", **link_kwargs)
    if min_length is not None:
        out = tp.filter_stubs(out, min_length)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_particles(ax, frame, particles, labels=None):
    """
    Overlay ring fits, inner masks and centers on `frame`.
    labels : optional list (same order as particles) of text labels, e.g.
        trackpy particle IDs.
    """
    ax.imshow(frame, cmap="gray")
    theta = np.linspace(0, 2 * np.pi, 200)
    for k, p in enumerate(particles):
        ring, inner = p["ring"], p["inner"]
        ax.plot(ring["center_x"], ring["center_y"], "+", mec="tab:orange",
                markersize=14, markeredgewidth=2)
        ax.plot(ring["center_x"] + ring["radius"] * np.cos(theta),
                ring["center_y"] + ring["radius"] * np.sin(theta),
                "-", c="tab:orange", linewidth=1.5)
        if inner is not None:
            x_off, y_off = p["mask_offset"]
            mask = inner["mask"]
            yy, xx = np.mgrid[y_off:y_off + mask.shape[0], x_off:x_off + mask.shape[1]]
            ax.contour(xx, yy, mask, levels=[0.5], colors="tab:green", linewidths=1.5)
            ax.plot(inner["center_x"], inner["center_y"], "+", mec="tab:green",
                    markersize=14, markeredgewidth=2)
        if labels is not None:
            ax.text(ring["center_x"] + ring["radius"], ring["center_y"] - ring["radius"],
                    str(labels[k]), color="yellow", fontsize=10)
    ax.axis("off")
