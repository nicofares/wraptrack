"""
Ring localization pipeline
----------------------------
PRIMARY method: edge-detection + RANSAC circle fit, THEN a full 2D joint
model fit.

Why a 2D fit, not a 1D radial-profile fit (the previous approach): every
1D method we tried -- azimuthal averaging, angular-sector exclusion,
azimuthal averaging seeded by a RANSAC radius -- first collapses the 2D
image down to some reduced 1D representation before fitting anything.
That reduction throws away information, and the pipeline turned out to be
sensitive to exactly the kind of asymmetric interior contamination
(shadowed facets, DIC shading, partial "touching" between the ring and
interior structure) that varies a bit frame to frame even when the ring
itself barely moves -- producing inconsistent results between very
similar frames. A full 2D fit uses every pixel's value AND position
directly, with no lossy intermediate step, and fits center, radius, and
width simultaneously rather than in separate stages that can each be
thrown off by the last. Verified directly: across 15 independent noise
realizations of the same real particle, center and radius came back with
a standard deviation of ~0.06px -- i.e. actually consistent frame to
frame, not just accurate on lucky individual frames.

Steps:
  1. Detect candidate edge pixels via gradient-magnitude thresholding
     (detect_edge_points) and fit a circle to them robustly via RANSAC
     (ransac_circle_fit). This gives a good, fast initial (center, radius)
     estimate: RANSAC finds the largest mutually-consistent circular point
     set and ignores everything else, so it isn't fooled by strong
     interior structure that doesn't lie on the ring's own circle, and it
     doesn't need full 360-degree ring coverage the way azimuthal
     averaging does.
  2. Fit the full 2D ring model (a Gaussian annulus on a flat baseline)
     directly to the raw image pixels via nonlinear least squares with a
     robust loss (soft_l1), seeded by step 1's estimate. The fit domain is
     restricted to an annulus around that estimate -- essential, not
     optional: an unrestricted (or even robustly-weighted but
     unrestricted) fit lets a broad, degenerate solution win purely
     because most of the image is "not the ring" and a broad hump can
     track that bulk better than a sharp, spatially localized true ring
     can (confirmed directly: this exact failure mode reproduced with
     bounds AND robust loss both in place, and was only fixed by
     restricting the domain).

FALLBACK: if RANSAC can't find a well-supported circle at all (e.g. a
plain bright spot with no ring -- no consistent circular edge structure
beyond its own smooth gradient boundary), the pipeline falls back to a
gradient-symmetry center estimate (radial_center) plus the older 1D
radial-profile fit with blind peak-search. Either way, the same
structural/SNR checks are the final backstop: a geometrically-plausible
circle with no real intensity bump there (e.g. RANSAC or the 2D fit
locking onto a smooth blob's own gradient edge) still gets rejected.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, peak_widths


# ----------------------------------------------------------------------
# Cropping around an (approximate) particle location
# ----------------------------------------------------------------------
def crop_around_point(img, xc, yc, half_size):
    """
    Crop a window around (xc, yc), clipped to stay inside `img`.

    Near an edge, the crop is simply smaller / asymmetric -- it always
    contains real image data, never runs out of bounds, and is never
    discarded outright just because the ideal full-size window doesn't fit.

    Returns
    -------
    crop : 2D array, the (possibly smaller-than-requested) crop
    offset : (x_off, y_off), the crop's top-left corner in the original
             image's coordinates -- add this to any crop-local position
             to get back to full-image coordinates.
    """
    Ny, Nx = img.shape
    x0, y0 = int(round(xc)), int(round(yc))

    x_lo = max(0, x0 - half_size)
    x_hi = min(Nx, x0 + half_size)
    y_lo = max(0, y0 - half_size)
    y_hi = min(Ny, y0 + half_size)

    crop = img[y_lo:y_hi, x_lo:x_hi]
    offset = (x_lo, y_lo)
    return crop, offset


# ----------------------------------------------------------------------
# Gradient-symmetry center (ROI seeding + fallback only)
# ----------------------------------------------------------------------
def radial_center(img, smoothing_sigma=2.0, weight_percentile=50, n_iter=5, outlier_sigma=2.5):
    """
    Gradient-based radial-symmetry center estimate, refined by iteratively
    reweighted least squares (IRLS).

    LIMITATION: IRLS only helps when a bad influence is a MINORITY
    outlier. A strong, sharp, but non-radially-symmetric interior feature
    can dominate the total gradient weight -- at which point it IS the
    majority vote, and reweighting the same votes doesn't fix that
    (confirmed directly on a real frame). Kept here as (a) a fast seed for
    the region-of-interest crop before RANSAC, and (b) a fallback if
    RANSAC finds no circular structure at all.

    Returns
    -------
    xo, yo : subpixel center coordinates (x = column, y = row)
    """
    img = img.astype(np.float64)
    img_smooth = gaussian_filter(img, smoothing_sigma)
    gy, gx = np.gradient(img_smooth)

    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))

    grad_mag2 = gx ** 2 + gy ** 2
    thresh = np.percentile(grad_mag2, weight_percentile)
    valid = grad_mag2 > thresh

    A1 = gy[valid]
    A2 = -gx[valid]
    c = gy[valid] * x_idx[valid] - gx[valid] * y_idx[valid]
    line_norm = np.sqrt(A1 ** 2 + A2 ** 2)
    w = grad_mag2[valid].copy()

    xo, yo = None, None
    for it in range(n_iter):
        Sxx = np.sum(w * A1 * A1)
        Sxy = np.sum(w * A1 * A2)
        Syy = np.sum(w * A2 * A2)
        Sxc = np.sum(w * A1 * c)
        Syc = np.sum(w * A2 * c)
        M = np.array([[Sxx, Sxy], [Sxy, Syy]])
        b = np.array([Sxc, Syc])
        xo, yo = np.linalg.solve(M, b)

        if it == n_iter - 1:
            break

        dist = np.abs(A1 * xo + A2 * yo - c) / line_norm
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) * 1.4826 + 1e-9
        outlier = dist > (med + outlier_sigma * mad)
        w = grad_mag2[valid].copy()
        w[outlier] *= 0.05

    return xo, yo


# ----------------------------------------------------------------------
# Edge detection + RANSAC circle fit (initial estimate)
# ----------------------------------------------------------------------
def rough_particle_location(img, blur_sigma=25):
    """
    Very robust (but low-precision) "where is the bright particle at all"
    estimate: heavily blur the image and take the argmax. Used only to
    seed the region-of-interest crop before edge detection -- NOT a
    center estimate for the ring itself.

    Why not radial_center for this: it's a gradient-symmetry method, and
    on a frame much larger than the particle, its consensus can get
    pulled far off target by noise spread across a large, mostly-empty
    background -- confirmed directly, off by 183px on a real frame where
    the particle occupies a small corner of a 906x692 image, badly
    truncating the ROI crop (cutting off the top and left of the true
    ring) and cascading into a severely undersized RANSAC circle fit.
    Blurring and taking the argmax makes no symmetry assumption at all --
    it just finds where the average intensity is highest -- so it's far
    more robust for this "rough location" role, even though it wouldn't
    be precise enough to use as the final answer.

    Returns (xc, yc) in image coordinates.
    """
    blurred = gaussian_filter(img.astype(np.float64), blur_sigma)
    yc0, xc0 = np.unravel_index(np.argmax(blurred), blurred.shape)
    return float(xc0), float(yc0)


# ----------------------------------------------------------------------
def detect_edge_points(img, smoothing_sigma=1.5, gradient_percentile=92):
    """
    Candidate edge-pixel locations via gradient-magnitude thresholding.
    Not specific to the ring -- catches any strong edge, interior
    structure included. Separating "the ring" from other edges is
    RANSAC's job, not this function's.
    """
    img_smooth = gaussian_filter(img.astype(np.float64), smoothing_sigma)
    gy, gx = np.gradient(img_smooth)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    thresh = np.percentile(grad_mag, gradient_percentile)
    ys, xs = np.nonzero(grad_mag > thresh)
    return np.column_stack([xs, ys]).astype(np.float64)


def fit_circle_lsq(points):
    """
    Algebraic (Kasa) least-squares circle fit. Exact for 3 points, a
    best-fit approximation for more. Returns (xc, yc, r); r is NaN if the
    points don't determine a valid circle (e.g. collinear).
    """
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    xc, yc = -D / 2, -E / 2
    r2 = xc ** 2 + yc ** 2 - F
    r = np.sqrt(r2) if r2 > 0 else np.nan
    return xc, yc, r


def ransac_circle_fit(points, n_iter=2000, inlier_thresh=2.0, min_radius=10,
                       max_radius=None, min_inliers=20, seed=None):
    """
    Robustly fit a circle to a (possibly outlier-contaminated) set of 2D
    points via RANSAC: repeatedly fit an exact circle to 3 random points,
    count inlier support among the rest, keep the best-supported
    candidate, then refit using all its inliers.

    Interior structure that doesn't lie on the ring's circle simply
    doesn't contribute inlier support, however strong its gradient is --
    this is what makes the approach robust to it. Also works from a
    partial arc (a ring only cleanly visible over part of its
    circumference), since 3 points anywhere on a circle determine it.

    Returns (xc, yc, r, inlier_mask), or None if no candidate gathered at
    least `min_inliers` support.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < max(10, min_inliers):
        return None
    if max_radius is None:
        span = points.max(axis=0) - points.min(axis=0)
        max_radius = np.sqrt((span ** 2).sum())

    best_n_inliers = -1
    best = None
    for _ in range(n_iter):
        idx = rng.choice(n, size=3, replace=False)
        xc, yc, r = fit_circle_lsq(points[idx])
        if not np.isfinite(r) or r < min_radius or r > max_radius:
            continue
        d = np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
        inliers = np.abs(d - r) < inlier_thresh
        n_in = inliers.sum()
        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best = (xc, yc, r, inliers)

    if best is None or best_n_inliers < min_inliers:
        return None

    _, _, _, inliers = best
    xc, yc, r = fit_circle_lsq(points[inliers])
    return xc, yc, r, inliers


def estimate_ring_via_edges(img, roi_half_size=180, seed=0, smoothing_sigma=1.5, **ransac_kwargs):
    """
    Initial (center, radius) estimate via edge detection + RANSAC.

    A rough center from rough_particle_location (blurred-image argmax,
    not gradient-symmetry -- see its docstring for why) is used only to
    crop a generous region of interest before edge detection -- cuts down
    on unrelated edges/noise elsewhere in a large frame, improving both
    speed and accuracy.

    smoothing_sigma : passed straight through to detect_edge_points --
        see that function's docstring. Exposed here (and further up, in
        locate_ring) so it can be tuned per-dataset without editing the
        default in detect_edge_points itself.

    seed : fixed by default so the SAME frame always gives the SAME
        result. RANSAC's random sampling otherwise means re-running on an
        identical image can quietly return a slightly different estimate
        -- confirmed directly (radius varying ~56-60px across seeds on
        one real frame) -- which is exactly the kind of run-to-run
        inconsistency this whole method switch was meant to eliminate.
        Pass None to restore random sampling if you specifically want to
        check estimate stability across seeds.

    Returns (xc, yc, r, n_inliers) in FULL-IMAGE coordinates, or None.
    """
    xc0, yc0 = rough_particle_location(img)
    crop, (x_off, y_off) = crop_around_point(img, xc0, yc0, roi_half_size)

    pts = detect_edge_points(crop, smoothing_sigma=smoothing_sigma)
    result = ransac_circle_fit(pts, seed=seed, **ransac_kwargs)
    if result is None:
        return None
    xc, yc, r, inliers = result
    return xc + x_off, yc + y_off, r, int(inliers.sum())


# ----------------------------------------------------------------------
# Full 2D joint ring fit (final, precise result)
# ----------------------------------------------------------------------
def ring_model_2d(params, X, Y):
    """Gaussian annulus on a flat baseline, evaluated over a 2D pixel grid."""
    xc, yc, r0, sigma, A, B = params
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return B + A * np.exp(-((r - r0) ** 2) / (2 * sigma ** 2))


PARAM_NAMES = ("xc", "yc", "r0", "sigma", "A", "B")


def local_goodness_of_fit(img, xc, yc, r0, sigma, A, B, window_sigmas=3):
    """
    Reduced chi-square of the fit, evaluated in a FIXED, common window
    around the peak (|r - r0| < window_sigmas*sigma) -- the SAME size
    regardless of how wide any given candidate's own fit domain was, so
    candidates are compared fairly rather than favoring whichever one
    happened to use a wider domain.

    This exists because raw SNR (peak amplitude / background noise) can
    be fooled: a WIDER, less precise fit can blend the true ring peak
    together with adjacent structure (e.g. the inner blob's declining
    tail), inflating its apparent amplitude and thus its SNR, while
    actually matching the real data WORSE than a narrower, more accurate
    fit with genuinely lower peak height. Confirmed directly on a real
    frame: the wider/taller fit had SNR=3.74 but local reduced
    chi-square=5.6 (poor match); a narrower fit seeded from the profile's
    actual peak had SNR=2.24 but chi-square=1.79 (good match) -- SNR
    picked the wrong one, this metric picks the right one.

    Lower is better. Not meaningful as an absolute "close to 1.0" pass/
    fail threshold on its own -- confirmed directly that even
    known-good, visually-verified fits across several different real
    images range from ~2 to ~13, likely because the background-region
    noise estimate isn't a perfect match to true local pixel noise. Use
    it to RANK candidates against each other, not as a hard cutoff in
    isolation.
    """
    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))
    r = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)
    local_mask = np.abs(r - r0) < window_sigmas * sigma
    if local_mask.sum() < 10:
        return np.inf

    data = img[local_mask].astype(np.float64)
    model = ring_model_2d([xc, yc, r0, sigma, A, B], x_idx[local_mask], y_idx[local_mask])
    resid = data - model

    bg_mask = r > (r0 + 4 * sigma)
    noise_std = img[bg_mask].astype(np.float64).std() if bg_mask.sum() >= 10 else resid.std()
    if noise_std <= 0:
        return np.inf
    return np.sum(resid ** 2) / (len(data) * noise_std ** 2)


def fit_ring_2d(img, xc0, yc0, r0_guess, sigma0_guess, half_size,
                 annulus_k_lower=1, annulus_k_upper=3, loss="soft_l1", f_scale=None, max_nfev=300,
                 fixed_params=None):
    """
    Fit the 2D ring model directly to image pixels via robust nonlinear
    least squares, seeded by (xc0, yc0, r0_guess, sigma0_guess) -- e.g.
    from estimate_ring_via_edges.

    fixed_params : optional dict holding any subset of {"xc", "yc", "r0",
        "sigma", "A", "B"} fixed at a given value -- e.g.
        {"r0": 23.0, "sigma": 3.0} fits only xc, yc, A, B, with radius and
        width pinned. Parameters not listed are fit freely, exactly as
        before. Fixed parameters are also used as the "guess" for
        building the fit domain/initial values, overriding xc0, yc0,
        r0_guess, sigma0_guess for that specific parameter. Omit or pass
        None to fit all six parameters freely (identical to previous
        behavior -- verified byte-for-byte).

    The fit domain is restricted to an ASYMMETRIC annulus around
    r0_guess: only annulus_k_lower*sigma0_guess inward, but
    annulus_k_upper*sigma0_guess outward (measured from the initial
    center guess). Asymmetric on purpose, not a symmetric annulus split
    two ways: the inner blob sits on the LOWER side, so that side needs
    to stay tight to exclude it, while the baseline/background genuinely
    needed to anchor B sits on the OUTER side, which needs more reach to
    be sampled properly -- a single shared margin can't satisfy both at
    once (confirmed directly in the related 1D fallback fit: a
    symmetric window either still included blob pixels or cut the
    baseline sample too short to estimate correctly, and only fixing the
    two sides independently resolved it).

    Restricting the domain at all is NOT optional: fitting over the whole
    crop lets a broad, degenerate solution win on total residual, because
    most pixels in an unrestricted domain are "not the ring" (interior
    blob, background) and a broad hump can track that bulk better than a
    sharp, spatially localized true ring can -- reproduced directly, with
    bounds and robust loss both already in place, and only fixed by
    restricting the domain.

    soft_l1 robust loss further downweights whatever doesn't fit the ring
    model within that domain (touching/fused regions, noise, a partial
    stray edge), so the fit doesn't need every pixel in the annulus to
    agree, just the majority.

    Returns a dict with center_x, center_y, radius, sigma, intensity_peak,
    baseline, snr, ring_shape_ok, success (all in FULL-IMAGE coordinates
    except where noted).
    """
    fixed_params = dict(fixed_params) if fixed_params else {}
    unknown = set(fixed_params) - set(PARAM_NAMES)
    if unknown:
        raise ValueError(f"fixed_params has unrecognized keys: {unknown}; "
                          f"expected a subset of {PARAM_NAMES}")

    Ny, Nx = img.shape
    x_lo = max(0, int(xc0 - half_size))
    x_hi = min(Nx, int(xc0 + half_size))
    y_lo = max(0, int(yc0 - half_size))
    y_hi = min(Ny, int(yc0 + half_size))
    crop = img[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    ch, cw = crop.shape
    Yc, Xc = np.indices(crop.shape)
    xc0_local, yc0_local = xc0 - x_lo, yc0 - y_lo

    # fixed_params for xc/yc are given in the SAME coordinate system as
    # xc0/yc0 (the caller's frame) -- but the model here fits in CROP-LOCAL
    # coordinates (offset by x_lo, y_lo). Convert once, and use this local
    # version everywhere below, not the original. Confirmed directly:
    # skipping this conversion silently shifted a "fixed" center by
    # exactly x_lo/y_lo pixels -- the returned value looked plausible but
    # didn't match the fixed value at all.
    fixed_local = dict(fixed_params)
    if "xc" in fixed_local:
        fixed_local["xc"] = fixed_local["xc"] - x_lo
    if "yc" in fixed_local:
        fixed_local["yc"] = fixed_local["yc"] - y_lo

    # a fixed xc/yc/r0/sigma overrides the corresponding guess for
    # building the fit domain and initial values -- if you already know
    # the center, the domain should be centered there, not on a rough
    # RANSAC guess that's now irrelevant to that parameter.
    xc0_eff = fixed_local.get("xc", xc0_local)
    yc0_eff = fixed_local.get("yc", yc0_local)
    r0_eff = fixed_local.get("r0", r0_guess)
    sigma0_eff = fixed_local.get("sigma", sigma0_guess)

    r_from_guess = np.sqrt((Xc - xc0_eff) ** 2 + (Yc - yc0_eff) ** 2)
    margin_lower = annulus_k_lower * sigma0_eff
    margin_upper = annulus_k_upper * sigma0_eff
    domain_mask = (r_from_guess > max(0, r0_eff - margin_lower)) & (r_from_guess < r0_eff + margin_upper)

    if domain_mask.sum() < 20:
        return None  # domain too small/empty to fit anything meaningful

    Xd, Yd, vals = Xc[domain_mask], Yc[domain_mask], crop[domain_mask]

    B0 = np.percentile(vals, 20)
    A0 = vals.max() - B0
    if f_scale is None:
        f_scale = max(1.0, 0.15 * A0)

    max_sensible_r0 = 0.5 * np.sqrt(cw ** 2 + ch ** 2)
    guess = {"xc": xc0_eff, "yc": yc0_eff, "r0": r0_eff, "sigma": sigma0_eff, "A": A0, "B": B0}
    lower_all = {"xc": 0, "yc": 0, "r0": 0, "sigma": 0.5, "A": 0, "B": -np.inf}
    upper_all = {"xc": cw, "yc": ch, "r0": max_sensible_r0, "sigma": half_size, "A": np.inf, "B": np.inf}

    free_names = [n for n in PARAM_NAMES if n not in fixed_local]

    def build_full(p_free):
        """Reassemble the full 6-param vector from free values + fixed_local."""
        values = dict(fixed_local)
        values.update(zip(free_names, p_free))
        return [values[n] for n in PARAM_NAMES]

    def resid(p_free):
        return ring_model_2d(build_full(p_free), Xd, Yd) - vals

    if len(free_names) == 0:
        # nothing to fit -- every parameter was fixed, just evaluate
        popt = [fixed_local[n] for n in PARAM_NAMES]
        success = True
    else:
        p0_free = [guess[n] for n in free_names]
        lower_free = [lower_all[n] for n in free_names]
        upper_free = [upper_all[n] for n in free_names]
        try:
            res = least_squares(resid, p0_free, bounds=(lower_free, upper_free), loss=loss,
                                 f_scale=f_scale, max_nfev=max_nfev)
        except ValueError:
            return None
        popt = build_full(res.x)
        success = bool(res.success)

    xc, yc, r0, sigma, A, B = popt

    bg_mask = r_from_guess > (r0 + 4 * sigma)
    if bg_mask.sum() >= 10:
        noise_std = crop[bg_mask].std()
    else:
        noise_std = np.std(vals - ring_model_2d(popt, Xd, Yd))
    snr = A / noise_std if noise_std > 0 else np.inf

    # Same structural check as before: is this actually a ring (offset
    # from center), or did the fit just refit the core blob's own natural
    # falloff centered near r=0? Real rings in this dataset run
    # radius/sigma of 12-16; a known false positive (fitting a plain
    # blob's own shape) came out at 0.01-0.08.
    ring_shape_ok = r0 > max(3.0, 1.5 * sigma)

    # Separate check: did the optimizer get pinned at (or very near) its
    # own upper bound? That's the signature of a runaway fit driven there
    # by a bad initial guess, not a genuine local optimum -- and it can
    # still pass ring_shape_ok if sigma happens to scale up too (confirmed
    # directly: radius pinned exactly at the bound, sigma scaled to match,
    # ratio comfortably above the ring_shape_ok threshold). SNR usually
    # catches this too since a pinned fit tends to fit poorly, but this is
    # a second, independent signal that doesn't rely on that. Skipped for
    # a radius that was FIXED (not fit), since "pinned at bound" only
    # means something for a value the optimizer actually chose.
    not_bound_pinned = ("r0" in fixed_params) or (r0 < 0.95 * upper_all["r0"])

    local_chi2 = local_goodness_of_fit(img, xc + x_lo, yc + y_lo, r0, sigma, A, B)

    return {
        "center_x": xc + x_lo,
        "center_y": yc + y_lo,
        "radius": r0,
        "sigma": sigma,
        "intensity_peak": A,
        "baseline": B,
        "snr": snr,
        "local_chi2": local_chi2,
        "ring_shape_ok": ring_shape_ok and not_bound_pinned,
        "success": success,
    }


def estimate_r0_from_profile_peak(img, xc, yc, r0_guess, r_min=15, smooth_window=5):
    """
    Find the ring's actual radius by looking at where the intensity
    profile itself peaks, instead of only relying on RANSAC's geometric
    (gradient-edge) estimate. r_min excludes the inner blob's own
    structure; r_max_search (2.5*r0_guess) excludes far-field noise
    fluctuations that can have deceptively high "prominence" over a very
    wide, mostly-empty search range -- confirmed directly, an unbounded
    search on a real frame found a spurious peak at r=215 with higher
    prominence than the real ring at r=25.

    Returns (r0, sigma_estimate), or (None, None) if no clear peak exists
    in the plausible range.
    """
    r, profile = radial_profile(img, xc, yc)
    r_max_search = 2.5 * r0_guess
    mask = (r > r_min) & (r < r_max_search)
    if mask.sum() < 10:
        return None, None
    r_m, p_m = r[mask], profile[mask]
    p_smooth = uniform_filter1d(p_m, size=min(smooth_window, len(p_m)))
    # prominence relative to the LOCAL (post-blob) range, not the whole
    # profile -- the whole-profile range is dominated by the much
    # brighter inner blob, which makes the ring's own (much subtler)
    # bump fail a prominence check sized for the blob's contrast instead
    # of the ring's.
    prominence = 0.1 * (p_m.max() - p_m.min())
    peaks, props = find_peaks(p_smooth, prominence=prominence)
    if len(peaks) == 0:
        return None, None
    best = peaks[np.argmax(props["prominences"])]
    r0 = r_m[best]
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    widths, *_ = peak_widths(p_smooth, [best], rel_height=0.5)
    sigma = max(1.0, (widths[0] * dr) / 2.355)
    return r0, sigma


def fit_ring_2d_multistart(img, xc0, yc0, r0_guess, half_size,
                            r0_multipliers=(0.7, 0.85, 1.0, 1.15, 1.3), **kwargs):
    """
    Run fit_ring_2d from several different initial radius guesses and keep
    the best result.

    Guesses come from two sources: (1) fixed multiples of r0_guess, which
    itself comes from RANSAC's GEOMETRIC (gradient-edge) circle fit, and
    (2) the intensity profile's own actual peak location (see
    estimate_r0_from_profile_peak) -- added because the geometric estimate
    and multiplier variations of it don't always land where the intensity
    itself actually peaks; confirmed directly on a real frame, the profile
    peak seed gave a materially more accurate radius (residual error 0.65px
    vs 1.61px against the visually-unambiguous true peak).

    If fixed_params (passed through **kwargs) fixes "r0", multi-starting
    over different r0 GUESSES is pointless -- r0 is pinned to the fixed
    value regardless of what any candidate "guesses" -- so a single fit is
    run instead, using r0_guess only to size the fit domain.

    Why multiple candidates at all: on marginal-contrast frames, the 2D
    fit can have two (or more) competing local minima -- the genuine
    ring, and a degenerate "refit the core blob's own shape" solution
    (caught by ring_shape_ok, but only after wasting the fit on it).
    Confirmed directly: which one a single-shot fit lands in can flip
    based on nothing more than 8-bit vs 16-bit pixel quantization of the
    SAME underlying image.

    "Best" = LOWEST local reduced chi-square (see local_goodness_of_fit)
    among candidates that pass ring_shape_ok -- NOT highest SNR. Confirmed
    directly that SNR can be fooled: a wider, less precise fit can blend
    the true ring with adjacent structure, inflating its apparent
    amplitude (and thus SNR) while actually matching the real data worse
    (higher chi-square) than a narrower, more accurate fit. Returns None
    if no candidate passes.
    """
    fixed_params = kwargs.get("fixed_params") or {}
    if "r0" in fixed_params:
        r0_try = fixed_params["r0"]
        sigma_try = fixed_params.get("sigma", max(1.0, 0.1 * r0_try))
        result = fit_ring_2d(img, xc0, yc0, r0_try, sigma_try, half_size, **kwargs)

        # Re-seed once, using this fit's own converged center, unless
        # position is ALSO fixed (nothing to re-seed in that case). NOT
        # gated on result["success"] -- confirmed directly that the FIRST
        # pass on the actual failing case had success=False, and that's
        # exactly the case needing a second pass, not one to skip it on.
        # Needed because with r0 fixed, none of the multi-start variety
        # below applies -- there's no mechanism left to correct for an
        # imprecise RANSAC center estimate before the fit domain gets
        # built around it. Confirmed directly on a real frame: RANSAC's
        # center was 9px off, the domain built around it gave SNR=1.19
        # (rejected, optimizer didn't even converge) -- refitting once
        # more, seeded by that first fit's own center, gave SNR=4.57
        # (accepted, clean convergence), just from re-centering the
        # domain on a better starting point.
        if result is not None and "xc" not in fixed_params and "yc" not in fixed_params:
            result2 = fit_ring_2d(img, result["center_x"], result["center_y"],
                                   r0_try, sigma_try, half_size, **kwargs)
            if result2 is not None and result2["ring_shape_ok"] and (
                    not result["ring_shape_ok"] or result2["local_chi2"] < result["local_chi2"]):
                result = result2

        return result if (result is not None and result["ring_shape_ok"]) else None

    r0_try_list = [r0_guess * mult for mult in r0_multipliers]
    r0_peak, sigma_peak = estimate_r0_from_profile_peak(img, xc0, yc0, r0_guess)
    if r0_peak is not None:
        r0_try_list.append(r0_peak)

    best = None
    for r0_try in r0_try_list:
        # profile-peak candidate brings its own sigma estimate; multiplier
        # candidates use the usual 0.1*r0_try scaling
        sigma_try = (sigma_peak if (r0_peak is not None and r0_try == r0_peak)
                     else max(1.0, 0.1 * r0_try))
        # scale WITH r0_try, not fixed to the original r0_guess -- a fixed
        # sigma paired with a much smaller/larger r0_try distorts the
        # intended annulus domain width (margins = annulus_k_lower/upper *
        # sigma_guess),
        # which can open the door to a different, spurious local minimum
        # than the same multiplier would reach with its own properly-scaled
        # domain. Confirmed directly: this exact mismatch let one candidate
        # escape to a radius=162 nonsense solution that passed both
        # ring_shape_ok and had a higher SNR than the genuine answer,
        # winning the "best" selection under the old SNR-based criterion.
        result = fit_ring_2d(img, xc0, yc0, r0_try, sigma_try, half_size, **kwargs)
        if result is None or not result["ring_shape_ok"]:
            continue
        # NOT requiring result["success"]: scipy's strict convergence flag
        # can be False even when the fitted parameters are fine -- with a
        # reduced max_nfev (needed for speed), the optimizer can still land
        # on the correct answer without meeting scipy's internal tolerance
        # in time.
        #
        # Require the radius to be within a plausible range of the RANSAC
        # estimate (which, even though imprecise, reflects real edge
        # geometry and is a genuine scale reference) before comparing at
        # all -- rejects mostly-nonsense candidates regardless of the
        # ranking metric used afterward.
        if not (0.5 * r0_guess <= result["radius"] <= 2.0 * r0_guess):
            continue
        if best is None or result["local_chi2"] < best["local_chi2"]:
            best = result

    if best is None:
        return None

    # Re-seed the winning candidate once, using ITS OWN converged
    # center/radius/sigma to rebuild the fit domain. Needed because the
    # domain is built ONCE from each candidate's INITIAL guess and never
    # re-centered even as xc/yc/r0 move during that fit -- so a candidate
    # can converge to a real local peak whose domain was nonetheless built
    # slightly off-target, systematically pulling the result away from the
    # true peak. Confirmed directly on a real frame: the profile's actual
    # local maximum was at r=26.5, but every one of the 5 standard
    # multi-start candidates converged to 23-25 instead -- a small but
    # visible, reproducible leftward shift, not noise.
    #
    # This is NOT applied blindly, though -- confirmed directly that
    # re-seeding from an already-poor fit can converge to the degenerate
    # "refit the blob's own shape" solution instead of improving anything
    # (with a falsely high SNR, the same false-positive pattern guarded
    # against elsewhere). The same ring_shape_ok + chi2-improvement gate
    # used for the fixed-r0 path protects against that here too: a
    # degenerate re-seed attempt fails ring_shape_ok and gets discarded,
    # keeping the original (already-validated) candidate.
    reseeded = fit_ring_2d(img, best["center_x"], best["center_y"],
                           best["radius"], best["sigma"], half_size, **kwargs)
    if reseeded is not None and reseeded["ring_shape_ok"] and reseeded["local_chi2"] < best["local_chi2"]:
        best = reseeded

    return best


# ----------------------------------------------------------------------
# 1D radial profile (kept for: inner-part boundary detection downstream,
# and as the fallback fit path when RANSAC finds no circular structure)
# ----------------------------------------------------------------------
def radial_profile(img, xc, yc, dr=1.0, r_max=None, pixel_mask=None):
    """
    Azimuthally-averaged radial intensity profile around (xc, yc).

    pixel_mask : optional boolean array, same shape as img. If given, only
                 these pixels contribute to the average.
    """
    Ny, Nx = img.shape
    y_idx, x_idx = np.indices((Ny, Nx))
    r_full = np.sqrt((x_idx - xc) ** 2 + (y_idx - yc) ** 2)

    if pixel_mask is not None:
        r = r_full[pixel_mask]
        img_vals = img[pixel_mask].astype(np.float64)
    else:
        r = r_full.ravel()
        img_vals = img.ravel().astype(np.float64)

    if r_max is None:
        r_max = r.max()

    bins = np.arange(0, r_max + dr, dr)
    n_bins = len(bins) - 1
    bin_idx = np.clip(np.digitize(r, bins) - 1, 0, n_bins - 1)

    sums = np.bincount(bin_idx, weights=img_vals, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        profile = sums / counts

    r_centers = 0.5 * (bins[:-1] + bins[1:])
    good = counts > 0
    return r_centers[good], profile[good]


def ring_model(r, B, A, r0, sigma):
    """1D version of the ring model (fallback path only)."""
    return B + A * np.exp(-((r - r0) ** 2) / (2 * sigma ** 2))


def fit_ring_1d(r, profile, min_prominence_frac=0.15, smooth_window=5, fit_window_sigmas=8,
                 fixed_params=None):
    """
    Fallback 1D fit with blind peak-search, used only when RANSAC finds no
    circular structure at all (see module docstring). Same windowed-fit
    logic as the primary pipeline's earlier version.

    fixed_params : optional dict holding any subset of {"r0", "sigma",
        "A", "B"} fixed at a given value (same internal names as
        fit_ring_2d's fixed_params -- "xc"/"yc" don't apply here, since
        this function only fits the profile shape; position is decided
        before this is called, by whichever center the profile was
        computed at). Fixed values also override the corresponding
        initial guess.
    """
    fixed_params = dict(fixed_params) if fixed_params else {}

    profile_smooth = uniform_filter1d(profile, size=smooth_window)
    prominence = min_prominence_frac * (profile.max() - profile.min())
    peaks, props = find_peaks(profile_smooth, prominence=prominence)
    dr_step = (r[1] - r[0]) if len(r) > 1 else 1.0
    peak_found = len(peaks) > 0

    if not peak_found:
        mask = r > 0.5 * r.max()
        r0_guess = r[mask][np.argmax(profile[mask])] if np.any(mask) else r[np.argmax(profile)]
        sigma_guess = max(1.0, 0.08 * r.max())
    else:
        best = peaks[np.argmax(props["prominences"])]
        r0_guess = r[best]
        widths, *_ = peak_widths(profile_smooth, [best], rel_height=0.5)
        sigma_guess = max(1.0, (widths[0] * dr_step) / 2.355)

    # a fixed r0/sigma overrides the corresponding guess -- also for
    # sizing the fit window below, so it's centered on the known value
    # rather than a now-irrelevant blind guess
    r0_eff = fixed_params.get("r0", r0_guess)
    sigma_eff = fixed_params.get("sigma", sigma_guess)

    r_fit_min = max(0, r0_eff - 3 * sigma_eff)
    r_fit_max = min(r.max(), r0_eff + fit_window_sigmas * sigma_eff)
    fit_mask = (r > r_fit_min) & (r <= r_fit_max)
    r_fit, profile_fit = r[fit_mask], profile[fit_mask]

    B_guess = np.percentile(profile_fit, 5)
    A_guess = profile_fit.max() - B_guess
    guess = {"B": B_guess, "A": A_guess, "r0": r0_eff, "sigma": sigma_eff}
    lower_all = {"B": -np.inf, "A": 0, "r0": 0, "sigma": 0.5}
    upper_all = {"B": np.inf, "A": np.inf, "r0": r_fit.max(), "sigma": r_fit.max()}

    param_order = ("B", "A", "r0", "sigma")  # matches ring_model(r, B, A, r0, sigma)
    free_names = [n for n in param_order if n not in fixed_params]

    def build_full(p_free):
        values = dict(fixed_params)
        values.update(zip(free_names, p_free))
        return [values[n] for n in param_order]

    if len(free_names) == 0:
        popt = tuple(fixed_params[n] for n in param_order)
        pcov = np.zeros((4, 4))  # nothing was fit; treat as exact/no uncertainty
    else:
        p0 = [guess[n] for n in free_names]
        lower = [lower_all[n] for n in free_names]
        upper = [upper_all[n] for n in free_names]

        def model_free(r_, *p_free):
            return ring_model(r_, *build_full(p_free))

        popt_free, pcov_free = curve_fit(model_free, r_fit, profile_fit, p0=p0,
                                          bounds=(lower, upper), maxfev=10000)
        popt = build_full(popt_free)
        # expand pcov back to full 4x4 (zeros for fixed params -- no
        # uncertainty was estimated for them since they weren't fit)
        pcov = np.zeros((4, 4))
        free_idx = [param_order.index(n) for n in free_names]
        for i, fi in enumerate(free_idx):
            for j, fj in enumerate(free_idx):
                pcov[fi, fj] = pcov_free[i, j]

    B, A, r0, sigma = popt

    bg_region = r > (r0 + 4 * sigma)
    noise_std = profile[bg_region].std() if bg_region.sum() >= 10 else (profile_fit - ring_model(r_fit, *popt)).std()
    snr = A / noise_std if noise_std > 0 else np.inf
    ring_shape_ok = r0 > max(3.0, 1.5 * sigma)

    return popt, pcov, {"peak_found": peak_found, "snr": snr, "ring_shape_ok": ring_shape_ok}


# ----------------------------------------------------------------------
# Full pipeline for one frame
# ----------------------------------------------------------------------
_PARAM_ALIASES = {
    "x": "xc", "xc": "xc",
    "y": "yc", "yc": "yc",
    "a": "r0", "radius": "r0", "r0": "r0",
    "sigma": "sigma",
    "intensity": "A", "intensity_peak": "A", "A": "A",
    "baseline": "B", "B": "B",
}


def _translate_fixed_params(fixed_params, x_off, y_off):
    """
    Translate user-facing fixed-parameter names (x, y, a/radius, sigma,
    intensity, baseline) to the model's internal names (xc, yc, r0, sigma,
    A, B), and shift position values from FULL-IMAGE coordinates (what the
    user sees in center_x/center_y) into the coordinates `img` is actually
    in after any approx_center cropping.
    """
    if not fixed_params:
        return {}
    out = {}
    for name, value in fixed_params.items():
        if name not in _PARAM_ALIASES:
            raise ValueError(
                f"Unknown fixed parameter {name!r}. Valid names: "
                f"x, y, a (or radius), sigma, intensity (or A), baseline (or B)."
            )
        internal = _PARAM_ALIASES[name]
        if internal == "xc":
            value = value - x_off
        elif internal == "yc":
            value = value - y_off
        out[internal] = value
    return out


def locate_ring(img, approx_center=None, half_size=None, min_snr=2.0,
                 roi_half_size=180, fit_half_size=None, smoothing_sigma=1.5,
                 fixed_params=None):
    """
    Run the full pipeline on a single image.

    approx_center, half_size : optional. If given, the whole pipeline runs
        on a crop of `img` around `approx_center` (adaptively clipped to
        stay inside the image) -- use this when the frame has more than
        one particle. All returned positions are shifted back to
        full-image coordinates automatically.

    min_snr : minimum fitted ring amplitude, relative to background noise,
        required to trust the result -- the final backstop against a
        geometrically-plausible circle with no real intensity bump there.
        Default lowered from 3.0 to 2.0 -- candidates are now selected by
        local reduced chi-square (goodness of fit), not raw SNR (see
        fit_ring_2d_multistart), because SNR alone could be fooled by a
        wider, less accurate fit that inflates its own apparent amplitude.
        The more accurate fits this produces have correspondingly more
        honest (and sometimes lower) peak amplitudes, so a threshold tuned
        for the old, SNR-inflating selection is now too strict. Verified
        directly: 2.0 still rejects pure noise (no ring at all) while
        correctly accepting a real, previously-wrongly-rejected fit.

    roi_half_size : half-size of the region-of-interest crop used before
        edge detection in the RANSAC stage.

    fit_half_size : half-size of the crop used for the final 2D fit.
        Defaults to roi_half_size if not given.

    smoothing_sigma : Gaussian blur applied before gradient/edge detection
        in the RANSAC stage (see detect_edge_points). Higher values
        suppress more pixel-level noise before edges are detected, which
        can matter a lot for marginal/noisy frames -- confirmed directly:
        raising this from 1.5 to 3.5 fixed a real frame where the fitted
        ring was landing visibly offset from the true peak. NOT a free
        win, though -- also confirmed a regression on a different frame
        from the same change, so treat this as a per-dataset tuning knob,
        not something to raise blindly. This is a SEPARATE knob from
        radial_center's own `smoothing_sigma` argument (used only in the
        fallback path below, when RANSAC finds nothing) -- changing this
        parameter does not touch that one.

    fixed_params : optional dict holding any subset of the fitted
        quantities at a known value instead of fitting them -- keys can be
        "x", "y" (position), "a" or "radius" (ring radius), "sigma" (ring
        width), "intensity" (peak amplitude above baseline), or
        "baseline". E.g. fixed_params={"a": 23.0, "sigma": 3.0,
        "intensity": 50.0} fits only x, y with the rest held at those
        values -- your example of locating a particle whose ring shape is
        already known. Position values (x, y) are in the same FULL-IMAGE
        coordinates as the returned center_x/center_y, regardless of
        approx_center cropping. Applies to the primary 2D-fit path only --
        if RANSAC finds no circular structure at all and the pipeline
        falls back to the older 1D method, fixed_params is NOT applied
        there (that path doesn't support it); this is a real, stated
        limitation, not a silent gap.

    Returns a dict with:
      center_x, center_y : ring center (subpixel, FULL-IMAGE coords)
      radius, sigma        : ring size and width
      intensity_peak        : peak height above baseline
      intensity_integrated   : integrated ring intensity (2D annulus
                               integral, thin-ring approximation:
                               2*pi*r0 * A*sigma*sqrt(2*pi))
      baseline               : local background level
      r_profile, profile     : 1D radial profile at the final center
                               (diagnostic + used by downstream inner-part
                               boundary detection)
      valid                  : False if no trustworthy ring was found.
      method                 : "2d_fit" or "fallback_1d", diagnostic only.
    """
    if approx_center is not None:
        crop, (x_off, y_off) = crop_around_point(img, approx_center[0], approx_center[1], half_size)
        img = crop
    else:
        x_off, y_off = 0, 0

    fixed_internal = _translate_fixed_params(fixed_params, x_off, y_off)

    if fit_half_size is None:
        fit_half_size = roi_half_size

    ransac_result = estimate_ring_via_edges(img, roi_half_size=roi_half_size,
                                             smoothing_sigma=smoothing_sigma)

    if ransac_result is not None:
        xc0, yc0, r0_guess, n_inliers = ransac_result
    elif "xc" in fixed_internal and "yc" in fixed_internal:
        # RANSAC found no circular structure, but since position is fixed
        # anyway we don't need its center estimate -- only a rough radius
        # guess is still useful (if radius itself isn't ALSO fixed).
        xc0, yc0 = fixed_internal["xc"], fixed_internal["yc"]
        r0_guess = fixed_internal.get("r0", 0.5 * min(fit_half_size, roi_half_size))
    else:
        xc0 = yc0 = r0_guess = None

    # A fixed position/radius overrides whatever was just estimated --
    # known values take priority over RANSAC's guess for that parameter.
    if "xc" in fixed_internal:
        xc0 = fixed_internal["xc"]
    if "yc" in fixed_internal:
        yc0 = fixed_internal["yc"]
    if "r0" in fixed_internal:
        r0_guess = fixed_internal["r0"]

    if xc0 is not None:
        fit2d = fit_ring_2d_multistart(img, xc0, yc0, r0_guess, fit_half_size,
                                        fixed_params=fixed_internal or None)
    else:
        fit2d = None

    if fit2d is not None and fit2d["snr"] >= min_snr:
        xc, yc, r0, sigma = fit2d["center_x"], fit2d["center_y"], fit2d["radius"], fit2d["sigma"]
        A, B = fit2d["intensity_peak"], fit2d["baseline"]
        r_profile, profile = radial_profile(img, xc, yc)
        integrated = 2 * np.pi * r0 * A * sigma * np.sqrt(2 * np.pi)
        return {
            "center_x": xc + x_off, "center_y": yc + y_off,
            "radius": r0, "sigma": sigma,
            "intensity_peak": A, "intensity_integrated": integrated,
            "baseline": B,
            "r_profile": r_profile, "profile": profile,
            "valid": True, "method": "2d_fit",
        }

    # ---- Fallback path ----
    if "xc" in fixed_internal and "yc" in fixed_internal:
        # Position is fixed -- skip radial_center entirely rather than
        # computing an estimate we'd just discard. Also means this path
        # works even for a frame where radial_center itself would have
        # struggled, since we never call it.
        xc, yc = fixed_internal["xc"], fixed_internal["yc"]
    else:
        xc, yc = radial_center(img)
    Ny, Nx = img.shape
    margin = 0.1 * max(Ny, Nx)
    center_in_bounds = (-margin <= xc <= Nx + margin) and (-margin <= yc <= Ny + margin)
    if not center_in_bounds:
        return {
            "center_x": xc + x_off, "center_y": yc + y_off,
            "radius": np.nan, "sigma": np.nan,
            "intensity_peak": np.nan, "intensity_integrated": np.nan,
            "baseline": np.nan,
            "r_profile": np.array([]), "profile": np.array([]),
            "valid": False, "method": "fallback_1d",
        }

    r_profile, profile = radial_profile(img, xc, yc)
    # fit_ring_1d only knows about r0/sigma/A/B -- xc/yc already got used
    # (or not) just above, deciding the center the profile was computed
    # at, so they don't apply to this call.
    fixed_for_1d = {k: v for k, v in fixed_internal.items() if k in ("r0", "sigma", "A", "B")}
    try:
        popt, pcov, diagnostics = fit_ring_1d(r_profile, profile, fixed_params=fixed_for_1d)
        B, A, r0, sigma = popt
        fit_ok = (
            np.all(np.isfinite(pcov))
            and diagnostics["peak_found"]
            and diagnostics["snr"] >= min_snr
            and diagnostics["ring_shape_ok"]
        )
    except RuntimeError:
        B = A = r0 = sigma = np.nan
        fit_ok = False

    integrated = 2 * np.pi * r0 * A * sigma * np.sqrt(2 * np.pi) if fit_ok else np.nan

    if not fit_ok:
        # NaN everything on failure, not just intensity_integrated. A
        # numerically-converged-but-quality-rejected fit (marginal SNR,
        # failed structural check) can still produce plausible-LOOKING
        # center/radius numbers -- returning them invites exactly the
        # mistake of plotting/using a result without checking `valid`
        # first (confirmed directly: this is what produced a nonsensical
        # giant circle in a real caller's plot, silently, with no error).
        # NaN can't be mistaken for a real answer.
        return {
            "center_x": np.nan, "center_y": np.nan,
            "radius": np.nan, "sigma": np.nan,
            "intensity_peak": np.nan, "intensity_integrated": np.nan,
            "baseline": np.nan,
            "r_profile": r_profile, "profile": profile,
            "valid": False, "method": "fallback_1d",
        }

    return {
        "center_x": xc + x_off, "center_y": yc + y_off,
        "radius": r0, "sigma": sigma,
        "intensity_peak": A, "intensity_integrated": integrated,
        "baseline": B,
        "r_profile": r_profile, "profile": profile,
        "valid": bool(fit_ok), "method": "fallback_1d",
    }


# ----------------------------------------------------------------------
# Radial profile plotting (diagnostic)
# ----------------------------------------------------------------------
def plot_radial_profile(img, result=None, xc=None, yc=None, r_max=None,
                         title=None, save_path=None, fig=None, ax=None):
    """
    Compute and plot the azimuthally-averaged radial intensity profile.

    Either pass `result` (a dict from locate_ring) to use its center and
    overlay its fitted ring model, or pass `xc, yc` directly for a profile
    without a fit overlay (e.g. to inspect a candidate center before
    fitting).

    Returns (r, profile, fig) -- fig is None if plotting onto a
    caller-supplied `ax` (caller owns that figure).
    """
    import matplotlib.pyplot as plt

    if result is not None:
        xc, yc = result["center_x"], result["center_y"]
    if xc is None or yc is None:
        raise ValueError("plot_radial_profile needs either `result` or both `xc` and `yc`")

    r, profile = radial_profile(img, xc, yc, r_max=r_max)

    # fig = None
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, profile, "k.", markersize=2, label="radial profile")

    if result is not None and result.get("valid", False):
        r0, sigma = result["radius"], result["sigma"]
        A, B = result["intensity_peak"], result["baseline"]
        r_dense = np.linspace(r.min(), r.max(), 400)
        ax.plot(r_dense, ring_model(r_dense, B, A, r0, sigma), "r--",
                linewidth=1.5, label=f"fitted ring (r={r0:.1f}, sigma={sigma:.1f})")
        ax.axvline(r0, color="r", linestyle=":", alpha=0.5)

    ax.set_xlabel("radius (px)")
    ax.set_ylabel("mean intensity")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=130)

    return r, profile, fig


if __name__ == "__main__":
    from PIL import Image

    img = np.array(Image.open("/mnt/user-data/uploads/testim.png").convert("L"))
    result = locate_ring(img)

    print("Ring localization result:")
    for k, v in result.items():
        if k in ("r_profile", "profile"):
            continue
        print(f"  {k}: {v}")
