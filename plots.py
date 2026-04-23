"""
Publication-quality plots for the NaN-extrapolation step in the destriping pipeline.
Example dataset: cell_10_CDI_3175974.npy

Four functions, each producing one self-contained figure:
  1. plot_extrapolation_stages   — step-by-step illustration of the full pipeline
  2. plot_blend_factor           — visualise the distance-based blending mechanism
  3. plot_degree_comparison      — justify the polynomial degree=2 choice
  4. plot_smooth_comparison      — justify the Gaussian sigma=2 choice (2D + cross-section)
"""

import copy
import os

import cmocean
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------------------------
# Paths & style constants
# ---------------------------------------------------------------------------

RASTER_PATH   = "rasters/cell_10_CDI_3175974.npy"
RASTER_PATH_2 = "rasters/cell_12_CDI_3466114.npy"
SAVE_DIR      = "temp/extrapolation_plots"

FS       = 15   # body / axis labels
FS_TITLE = 16   # panel titles
DPI      = 200

CMAP_BATH = cmocean.cm.deep
NAN_COLOR = "#cccccc"
HIGHLIGHT_COLOR = "#e6771e"   # orange border for the chosen parameter
DETREND_SIGMA_CHOSEN = 6    # chosen Gaussian detrending sigma (pixels = 60 m at 20 m/px)
NOTCH_WIDTH_CHOSEN  = 5     # chosen notch half-width (pixels in padded FFT)
NOTCH_CENTER_SIZE   = 15    # DC-exclusion radius (matches _find_angle in destripeClass)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_example():
    """Load cell_10_CDI_3175974, demean, and return (demeaned_raster, mean_value)."""
    raster = np.load(RASTER_PATH)
    mean_val = np.nanmean(raster)
    return raster - mean_val, mean_val


def _poly_surface(raster_dm, degree):
    """Fit a polynomial surface of `degree` to valid pixels; evaluate over full grid."""
    h, w = raster_dm.shape
    valid_mask = ~np.isnan(raster_dm)
    y_v, x_v = np.where(valid_mask)
    z_v = raster_dm[valid_mask]

    poly  = PolynomialFeatures(degree=degree, include_bias=True)
    X_fit = poly.fit_transform(np.column_stack([x_v, y_v]))
    model = LinearRegression().fit(X_fit, z_v)

    yg, xg     = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    X_full     = poly.transform(np.column_stack([xg.ravel(), yg.ravel()]))
    surface    = model.predict(X_full).reshape(h, w)
    return surface


def _nn_fill(raster_dm):
    """Nearest-neighbour fill: every NaN cell takes the value of its closest valid pixel."""
    valid_mask = ~np.isnan(raster_dm)
    indices    = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    return raster_dm[tuple(indices)]


def _blend(raster_dm, nn_filled, poly_surface, blend_dist=15):
    """
    Blend NN fill (near valid data) with polynomial surface (far from data).
    blend_factor = clip(dist / blend_dist, 0, 1)²
    Returns (blended array, blend_factor map).
    """
    valid_mask     = ~np.isnan(raster_dm)
    dist           = distance_transform_edt(~valid_mask)
    blend_factor   = np.clip(dist / blend_dist, 0, 1) ** 2
    blended        = nn_filled * (1 - blend_factor) + poly_surface * blend_factor
    blended[valid_mask] = raster_dm[valid_mask]
    return blended, blend_factor, dist


def _smooth_iters(blended, raster_dm, sigma, n_iter=3):
    """Apply `n_iter` passes of Gaussian smoothing (sigma), restoring valid data after each."""
    valid_mask = ~np.isnan(raster_dm)
    result     = blended.copy()
    for _ in range(n_iter):
        if sigma > 0:
            result = ndimage.gaussian_filter(result, sigma=sigma)
        result[valid_mask] = raster_dm[valid_mask]
    return result


def _imshow_nan(ax, data, nan_mask, cmap=CMAP_BATH, **kwargs):
    """imshow with NaN pixels rendered as NAN_COLOR."""
    cm = copy.copy(cmap)
    cm.set_bad(NAN_COLOR)
    masked = np.ma.masked_where(nan_mask, data)
    return ax.imshow(masked, cmap=cm, origin="upper", **kwargs)


def _vrange(raster_dm, mean_val):
    """Colour limits derived from valid data (original depth scale)."""
    return np.nanmin(raster_dm + mean_val), np.nanmax(raster_dm + mean_val)


def _best_cross_section_row(nan_mask):
    """
    Find the row with the clearest NaN-to-valid transition:
    largest minimum valid margin on both sides of the NaN segment.
    Falls back to the row with any NaN if nothing ideal is found.
    """
    best_row, best_score = None, -1
    for r in range(nan_mask.shape[0]):
        row = nan_mask[r]
        if not row.any() or (~row).any() is False:
            continue
        nan_pos     = np.where(row)[0]
        first_nan   = nan_pos[0]
        last_nan    = nan_pos[-1]
        score       = min(first_nan, nan_mask.shape[1] - 1 - last_nan)
        if score > best_score:
            best_score, best_row = score, r
    if best_row is None:
        rows = np.where(np.any(nan_mask, axis=1))[0]
        best_row = rows[len(rows) // 2] if len(rows) else nan_mask.shape[0] // 2
    return best_row


def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 1 — Extrapolation stages
# ---------------------------------------------------------------------------

def plot_extrapolation_stages(save_path=None):
    """
    Five-panel figure showing the NaN-extrapolation pipeline step by step:
      Original → NN fill → Polynomial surface → Blended → Smoothed (final).
    """
    raster_dm, mean_val = _load_example()
    nan_mask = np.isnan(raster_dm)
    vmin, vmax = _vrange(raster_dm, mean_val)

    poly    = _poly_surface(raster_dm, degree=2)
    nn      = _nn_fill(raster_dm)
    blended, _, _ = _blend(raster_dm, nn, poly)
    smoothed = _smooth_iters(blended, raster_dm, sigma=2, n_iter=3)

    stages = [
        (raster_dm + mean_val, "Original\n(NaN = survey gap)",      nan_mask),
        (nn        + mean_val, "Stage 1: nearest-neighbour\nfill",   np.zeros_like(nan_mask)),
        (poly      + mean_val, "Stage 2: polynomial trend\n(degree 2)", np.zeros_like(nan_mask)),
        (blended   + mean_val, "Stage 3: blended\n(NN ↔ polynomial)", np.zeros_like(nan_mask)),
        (smoothed  + mean_val, "Stage 4: smoothed\n(σ = 2,  3 passes)", np.zeros_like(nan_mask)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.91, top=0.88, bottom=0.06)

    for ax, (data, title, mask) in zip(axes, stages):
        im = _imshow_nan(ax, data, mask, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=FS_TITLE, pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.72])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Depth (m)", fontsize=FS)

    # NaN legend on first panel
    patch = mpatches.Patch(color=NAN_COLOR, label="No data (NaN)")
    axes[0].legend(handles=[patch], fontsize=FS - 2, loc="lower right",
                   framealpha=0.9, edgecolor="#aaaaaa")

    # Arrows between panels
    for ax in axes[:-1]:
        ax.annotate("", xy=(1.03, 0.5), xycoords="axes fraction",
                    xytext=(1.0, 0.5), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2 — Blend factor
# ---------------------------------------------------------------------------

def plot_blend_factor(save_path=None):
    """
    Four-panel figure explaining the distance-based blending mechanism:
      Distance map | Blend weight | NN fill | Blended result.
    Shows why a quadratic blend is used instead of a hard switch.
    """
    raster_dm, mean_val = _load_example()
    nan_mask   = np.isnan(raster_dm)
    valid_mask = ~nan_mask
    vmin, vmax = _vrange(raster_dm, mean_val)

    poly                    = _poly_surface(raster_dm, degree=2)
    nn                      = _nn_fill(raster_dm)
    blended, blend_factor, dist = _blend(raster_dm, nn, poly)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.10, left=0.03, right=0.97, top=0.88, bottom=0.06)

    # Panel 1 — distance from valid data
    im0 = axes[0].imshow(dist, cmap="YlOrRd", origin="upper")
    axes[0].set_title("Distance from\nvalid data (pixels)", fontsize=FS_TITLE)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("px", fontsize=FS - 1)

    # Panel 2 — blend factor
    im1 = axes[1].imshow(blend_factor, cmap="RdYlGn_r", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Blend weight\n(0 = NN,  1 = polynomial)", fontsize=FS_TITLE)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(["NN", "0.5", "Poly"])

    # Panel 3 — NN fill
    _imshow_nan(axes[2], nn + mean_val, nan_mask, vmin=vmin, vmax=vmax)
    axes[2].set_title("Nearest-neighbour fill\n(no blending)", fontsize=FS_TITLE)

    # Panel 4 — blended result
    im3 = _imshow_nan(axes[3], blended + mean_val, np.zeros_like(nan_mask), vmin=vmin, vmax=vmax)
    axes[3].set_title("Blended result\n(NN near data, poly far away)", fontsize=FS_TITLE)
    cb3 = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03)
    cb3.set_label("Depth (m)", fontsize=FS)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # Annotate the blend distance on the distance panel
    axes[0].contour(dist, levels=[15], colors=[HIGHLIGHT_COLOR], linewidths=1.5,
                    linestyles="--")
    axes[0].text(0.02, 0.03, "─ ─  blend radius (15 px)", transform=axes[0].transAxes,
                 fontsize=FS - 2, color=HIGHLIGHT_COLOR, va="bottom")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3 — Polynomial degree comparison
# ---------------------------------------------------------------------------

def plot_degree_comparison(save_path=None):
    """
    2×3 grid comparing polynomial degrees 1, 2, 3.
    Row 1: the polynomial trend surface.
    Row 2: the final extrapolation (blended + smoothed).
    Chosen degree (2) is highlighted with an orange border.
    """
    raster_dm, mean_val = _load_example()
    nan_mask = np.isnan(raster_dm)
    vmin, vmax = _vrange(raster_dm, mean_val)

    degrees = [1, 2, 3]
    col_labels = ["Degree 1  (planar)", "Degree 2  (quadratic)", "Degree 3  (cubic)"]

    nn      = _nn_fill(raster_dm)
    results = []
    for d in degrees:
        surf    = _poly_surface(raster_dm, d)
        blended, _, _ = _blend(raster_dm, nn, surf)
        final   = _smooth_iters(blended, raster_dm, sigma=2, n_iter=3)
        results.append((surf, final))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.04, hspace=0.10,
                        left=0.08, right=0.92, top=0.93, bottom=0.04)

    row_labels = ["Polynomial\ntrend surface", "Final\nextrapolation"]

    for col, ((surf, final), col_lbl) in enumerate(zip(results, col_labels)):
        # Surface row — no NaN mask (surface is defined everywhere)
        im = axes[0, col].imshow(surf + mean_val, cmap=CMAP_BATH,
                                 origin="upper", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(col_lbl, fontsize=FS_TITLE + 1, pad=8)

        # Final result row — show original NaN locations
        _imshow_nan(axes[1, col], final + mean_val, nan_mask, vmin=vmin, vmax=vmax)

        for row in range(2):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.80])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Depth (m)", fontsize=FS)

    # Row labels on left axis
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=FS_TITLE, labelpad=8)

    # Orange border on chosen degree (col 1 = degree 2)
    for row in range(2):
        for spine in axes[row, 1].spines.values():
            spine.set_edgecolor(HIGHLIGHT_COLOR)
            spine.set_linewidth(3)

    fig.text(0.5, 0.97, "Polynomial degree comparison  (chosen: degree 2)",
             ha="center", va="top", fontsize=FS_TITLE + 1, fontweight="bold")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 4 — Smoothing parameter comparison
# ---------------------------------------------------------------------------

def plot_smooth_comparison(save_path=None):
    """
    Two-part figure justifying the Gaussian smoothing parameter σ = 2:
      Top:    Four 2D panels for σ ∈ {0, 1, 2, 5} (dashed line = cross-section row).
      Bottom: 1D cross-section through the NaN boundary showing transition sharpness.
    The chosen σ (= 2) is drawn thicker in the cross-section.
    """
    raster_dm, mean_val = _load_example()
    nan_mask   = np.isnan(raster_dm)
    vmin, vmax = _vrange(raster_dm, mean_val)

    poly    = _poly_surface(raster_dm, degree=2)
    nn      = _nn_fill(raster_dm)
    blended, _, _ = _blend(raster_dm, nn, poly)

    sigmas = [0, 1, 2, 5]
    labels = ["σ = 0  (no smoothing)", "σ = 1", "σ = 2  (chosen)", "σ = 5"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    finals = [_smooth_iters(blended, raster_dm, sigma=s, n_iter=3) for s in sigmas]

    best_row = _best_cross_section_row(nan_mask)

    # ---- layout ----
    fig = plt.figure(figsize=(20, 9))
    gs  = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0],
                           hspace=0.12, wspace=0.06,
                           left=0.05, right=0.97, top=0.87, bottom=0.07)
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax_cs    = fig.add_subplot(gs[1, :])

    # ---- top panels ----
    for col, (final, lbl, color) in enumerate(zip(finals, labels, colors)):
        im = axes_top[col].imshow(final + mean_val, cmap=CMAP_BATH,
                                  origin="upper", vmin=vmin, vmax=vmax)
        axes_top[col].set_title(lbl, fontsize=FS_TITLE, pad=6, color=color if col != 2 else "#000000")
        axes_top[col].set_xticks([]); axes_top[col].set_yticks([])
        # Mark the cross-section row
        axes_top[col].axhline(best_row, color=color, lw=1.6, ls="--", alpha=0.85)

        # Orange border for chosen parameter
        if col == 2:
            for spine in axes_top[col].spines.values():
                spine.set_edgecolor(HIGHLIGHT_COLOR)
                spine.set_linewidth(3)

    cbar_ax = fig.add_axes([0.975, 0.55, 0.012, 0.35])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Depth (m)", fontsize=FS)

    # ---- cross-section ----
    x       = np.arange(raster_dm.shape[1])
    nan_row = nan_mask[best_row]

    # Shade NaN gaps
    in_nan, nan_start, first_shaded = False, 0, True
    for xi, is_nan in enumerate(nan_row):
        if is_nan and not in_nan:
            nan_start = xi; in_nan = True
        elif not is_nan and in_nan:
            ax_cs.axvspan(nan_start, xi, alpha=0.13, color="gray",
                          label="Survey gap (NaN)" if first_shaded else "")
            in_nan = False; first_shaded = False
    if in_nan:
        ax_cs.axvspan(nan_start, len(nan_row), alpha=0.13, color="gray")

    for final, lbl, color, sigma in zip(finals, labels, colors, sigmas):
        lw = 2.8 if sigma == 2 else 1.5
        ax_cs.plot(x, final[best_row] + mean_val, color=color, lw=lw,
                   label=lbl, zorder=3 if sigma == 2 else 2)

    ax_cs.set_xlabel("Column index", fontsize=FS)
    ax_cs.set_ylabel("Depth (m)", fontsize=FS)
    ax_cs.set_title(f"Cross-section at row {best_row}  "
                    "(dashed lines in upper panels mark this row)", fontsize=FS_TITLE)
    ax_cs.legend(fontsize=FS - 1, ncol=5, loc="best", framealpha=0.9)
    ax_cs.grid(alpha=0.3)
    ax_cs.set_xlim(0, raster_dm.shape[1] - 1)

    fig.suptitle("Gaussian smoothing parameter comparison  (chosen: σ = 2)",
                 fontsize=FS_TITLE + 1, fontweight="bold", y=0.98)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 5 — Blend distance comparison
# ---------------------------------------------------------------------------

def plot_blend_dist_comparison(save_path=None):
    """
    Three-row figure justifying the blend_dist = 15 px parameter:
      Row 1: Final extrapolated result for blend_dist ∈ {5, 15, 30, 60}.
      Row 2: Blend factor map (0 = NN, 1 = polynomial) for each value.
      Row 3: 1D cross-section through the NaN boundary.
    The chosen value (15) is highlighted with an orange border.
    """
    raster_dm, mean_val = _load_example()
    nan_mask   = np.isnan(raster_dm)
    vmin, vmax = _vrange(raster_dm, mean_val)

    poly = _poly_surface(raster_dm, degree=2)
    nn   = _nn_fill(raster_dm)

    blend_dists = [5, 15, 30, 60]
    labels      = ["d = 5 px", "d = 15 px  (chosen)", "d = 30 px", "d = 60 px"]
    colors      = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    chosen_col  = 1   # blend_dist = 15

    results, blends = [], []
    for bd in blend_dists:
        blended, blend_factor, _ = _blend(raster_dm, nn, poly, blend_dist=bd)
        results.append(_smooth_iters(blended, raster_dm, sigma=2, n_iter=3))
        blends.append(blend_factor)

    best_row = _best_cross_section_row(nan_mask)

    # ---- layout ----
    fig = plt.figure(figsize=(20, 13))
    gs  = fig.add_gridspec(3, 4, height_ratios=[1.4, 1.4, 1.0],
                           hspace=0.14, wspace=0.06,
                           left=0.05, right=0.95, top=0.87, bottom=0.06)
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(4)]
    axes_mid = [fig.add_subplot(gs[1, c]) for c in range(4)]
    ax_cs    = fig.add_subplot(gs[2, :])

    # ---- row 1: final result ----
    for col, (final, lbl, color) in enumerate(zip(results, labels, colors)):
        im_top = axes_top[col].imshow(final + mean_val, cmap=CMAP_BATH,
                                      origin="upper", vmin=vmin, vmax=vmax)
        axes_top[col].set_title(lbl, fontsize=FS_TITLE, pad=6)
        axes_top[col].set_xticks([]); axes_top[col].set_yticks([])
        axes_top[col].axhline(best_row, color=color, lw=1.6, ls="--", alpha=0.85)
        if col == chosen_col:
            for spine in axes_top[col].spines.values():
                spine.set_edgecolor(HIGHLIGHT_COLOR); spine.set_linewidth(3)

    cbar_ax_top = fig.add_axes([0.955, 0.56, 0.012, 0.28])
    cbar_top = fig.colorbar(im_top, cax=cbar_ax_top)
    cbar_top.set_label("Depth (m)", fontsize=FS)

    # ---- row 2: blend factor ----
    for col, (bf, color) in enumerate(zip(blends, colors)):
        im_mid = axes_mid[col].imshow(bf, cmap="RdYlGn_r", origin="upper", vmin=0, vmax=1)
        axes_mid[col].set_xticks([]); axes_mid[col].set_yticks([])
        axes_mid[col].axhline(best_row, color=color, lw=1.6, ls="--", alpha=0.85)
        if col == chosen_col:
            for spine in axes_mid[col].spines.values():
                spine.set_edgecolor(HIGHLIGHT_COLOR); spine.set_linewidth(3)

    cbar_ax_mid = fig.add_axes([0.955, 0.24, 0.012, 0.28])
    cbar_mid = fig.colorbar(im_mid, cax=cbar_ax_mid)
    cbar_mid.set_ticks([0, 0.5, 1])
    cbar_mid.set_ticklabels(["NN", "0.5", "Poly"])
    cbar_mid.set_label("Blend weight", fontsize=FS)

    axes_top[0].set_ylabel("Final\nextrapolation", fontsize=FS_TITLE, labelpad=8)
    axes_mid[0].set_ylabel("Blend factor\n(0 = NN,  1 = poly)", fontsize=FS_TITLE, labelpad=8)

    # ---- cross-section ----
    x       = np.arange(raster_dm.shape[1])
    nan_row = nan_mask[best_row]

    in_nan, nan_start, first_shaded = False, 0, True
    for xi, is_nan in enumerate(nan_row):
        if is_nan and not in_nan:
            nan_start = xi; in_nan = True
        elif not is_nan and in_nan:
            ax_cs.axvspan(nan_start, xi, alpha=0.13, color="gray",
                          label="Survey gap (NaN)" if first_shaded else "")
            in_nan = False; first_shaded = False
    if in_nan:
        ax_cs.axvspan(nan_start, len(nan_row), alpha=0.13, color="gray")

    for final, lbl, color, bd in zip(results, labels, colors, blend_dists):
        lw = 2.8 if bd == 15 else 1.5
        ax_cs.plot(x, final[best_row] + mean_val, color=color, lw=lw,
                   label=lbl, zorder=3 if bd == 15 else 2)

    ax_cs.set_xlabel("Column index", fontsize=FS)
    ax_cs.set_ylabel("Depth (m)", fontsize=FS)
    ax_cs.set_title(f"Cross-section at row {best_row}  "
                    "(dashed lines above mark this row)", fontsize=FS_TITLE)
    ax_cs.legend(fontsize=FS - 1, ncol=5, loc="best", framealpha=0.9)
    ax_cs.grid(alpha=0.3)
    ax_cs.set_xlim(0, raster_dm.shape[1] - 1)

    fig.suptitle("NN blend distance comparison  (chosen: d = 15 px)",
                 fontsize=FS_TITLE + 1, fontweight="bold", y=0.98)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Helpers for detrending plots
# ---------------------------------------------------------------------------

def _fill_raster():
    """Load the example raster, demean, and NaN-fill with the nearest_smooth pipeline.
    Returns (filled_dm, mean_val, nan_mask_original).
    """
    raster_dm, mean_val = _load_example()
    nan_mask = np.isnan(raster_dm)
    filled   = raster_dm.copy()
    while np.isnan(filled).any():
        valid_mask        = ~np.isnan(filled)
        surf              = _poly_surface(filled, degree=2)
        nn                = _nn_fill(filled)
        blended, _, _     = _blend(filled, nn, surf, blend_dist=15)
        result            = blended.copy()
        for _ in range(3):
            result = ndimage.gaussian_filter(result, sigma=2)
            result[valid_mask] = filled[valid_mask]
        filled = result
    return filled, mean_val, nan_mask


def _fill_raster_path(path):
    """NaN-fill pipeline identical to _fill_raster but for any raster file."""
    raster    = np.load(path)
    mean_val  = np.nanmean(raster)
    raster_dm = raster - mean_val
    nan_mask  = np.isnan(raster_dm)
    filled    = raster_dm.copy()
    while np.isnan(filled).any():
        valid_mask    = ~np.isnan(filled)
        surf          = _poly_surface(filled, degree=2)
        nn            = _nn_fill(filled)
        blended, _, _ = _blend(filled, nn, surf, blend_dist=15)
        result        = blended.copy()
        for _ in range(3):
            result = ndimage.gaussian_filter(result, sigma=2)
            result[valid_mask] = filled[valid_mask]
        filled = result
    return filled, mean_val, nan_mask


def _gaussian_detrend(filled, sigma):
    """Return (trend, residuals) for Gaussian detrending with given sigma."""
    trend = ndimage.gaussian_filter(filled, sigma=sigma)
    return trend, filled - trend


def _fft_log_amp(residuals):
    """Centred log-scale FFT amplitude (no padding — fast approximation for plots)."""
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(residuals))) + 1)


def _build_notch(shape, notch_width=5, center_size=15):
    """Diagonal-band mask in frequency space (same logic as Destriper._create_notch)."""
    h, w  = shape
    notch = np.zeros((h, w), dtype=float)
    for i in range(-notch_width, notch_width + 1):
        notch += np.eye(h, w, k=i)
    cy, cx = h // 2, w // 2
    notch[cy - center_size:cy + center_size, cx - center_size:cx + center_size] = 0
    return notch


def _find_stripe_angle(log_amp, step=1):
    """Scan 0–179° and return (best_angle, responses, relative_heights)."""
    notch  = _build_notch(log_amp.shape)
    angles = np.arange(0, 180, step)
    resp   = []
    for theta in angles:
        rot = ndimage.rotate(notch, float(theta), reshape=False)
        resp.append(np.sum(rot * log_amp) / (np.sum(rot) + 1e-12))
    resp = np.array(resp)
    rel  = np.array([
        resp[i] - np.mean(resp[np.roll(np.arange(len(resp)), -i + 4)[:8]])
        for i in range(len(resp))
    ])
    best = int(np.argmax(rel))
    return float(angles[best]), resp, rel


def _stripe_prominence(log_amp, angle_deg, notch_width=5, center_size=15):
    """
    Ratio of mean FFT amplitude inside the stripe band to the global mean.
    > 1 means the stripe band is brighter than average (energy concentrated in
    one direction), which indicates that stripes are detectable in the residuals.
    """
    h, w   = log_amp.shape
    notch  = _build_notch((h, w), notch_width, center_size)
    rot    = ndimage.rotate(notch, float(angle_deg), reshape=False) > 0.5
    cy, cx = h // 2, w // 2
    dc     = np.zeros((h, w), dtype=bool)
    dc[cy - center_size:cy + center_size, cx - center_size:cx + center_size] = True
    global_mean = np.mean(log_amp[~dc])
    stripe_mean = float(np.mean(log_amp[rot])) if rot.any() else 0.0
    return stripe_mean / (global_mean + 1e-12)


def _windowed_padded_fft(data):
    """
    2-D Hann-windowed, zero-padded FFT → log-scale amplitude.
    Windowing suppresses spectral leakage at image edges;
    zero-padding increases frequency resolution.
    """
    h, w   = data.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    pad    = max(h, w) // 2
    padded = np.pad(data * window, pad, mode="constant")
    F      = np.fft.fftshift(np.fft.fft2(padded))
    return np.log(np.abs(F) + 1)


# ---------------------------------------------------------------------------
# Figure 6 — Detrending stages (chosen sigma)
# ---------------------------------------------------------------------------

def plot_detrend_stages(save_path=None):
    """
    Three-panel figure illustrating the Gaussian detrending step (chosen σ):
      NaN-filled raster  |  Gaussian trend surface  |  Residuals.
    Residuals use a diverging colormap so stripe patterns are immediately visible.
    """
    filled, mean_val, nan_mask = _fill_raster()
    sigma = DETREND_SIGMA_CHOSEN
    trend, residuals = _gaussian_detrend(filled, sigma)

    vmin, vmax = np.nanmin(filled + mean_val), np.nanmax(filled + mean_val)
    res_lim    = float(np.max(np.abs(residuals)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.12, left=0.04, right=0.97, top=0.88, bottom=0.06)

    # Panel 1 — filled raster (survey gap still shown)
    _imshow_nan(axes[0], filled + mean_val, nan_mask, vmin=vmin, vmax=vmax)
    axes[0].set_title("NaN-filled raster\n(input to detrending)", fontsize=FS_TITLE, pad=8)
    patch = mpatches.Patch(color=NAN_COLOR, label="Survey gap (NaN)")
    axes[0].legend(handles=[patch], fontsize=FS - 2, loc="lower right",
                   framealpha=0.9, edgecolor="#aaaaaa")

    # Panel 2 — Gaussian trend surface
    im2 = axes[1].imshow(trend + mean_val, cmap=CMAP_BATH, origin="upper",
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Gaussian trend\n(σ = {sigma} px = {sigma * 20} m)",
                       fontsize=FS_TITLE, pad=8)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03).set_label("Depth (m)", fontsize=FS)

    # Panel 3 — residuals (diverging so stripes pop)
    im3 = axes[2].imshow(residuals, cmap="RdBu_r", origin="upper",
                          vmin=-res_lim, vmax=res_lim)
    axes[2].set_title("Residuals  (filled − trend)\nstripe pattern visible here",
                       fontsize=FS_TITLE, pad=8)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.03).set_label("Residual (m)", fontsize=FS)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Gaussian detrending pipeline  (σ = {sigma} px = {sigma * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 7 — Detrending sigma comparison
# ---------------------------------------------------------------------------

def plot_detrend_sigma_comparison(save_path=None):
    """
    Multi-row comparison of Gaussian detrending sigmas.

    Each row corresponds to one sigma value.
    Columns: Trend | FFT(trend, demeaned) | Residuals | FFT(residuals).

    FFTs use a 2-D Hann window + zero-padding to reduce spectral leakage.
    The trend is demeaned before its FFT so the DC origin does not saturate
    the colour scale.  Orange border marks the chosen sigma row.
    """
    print("  Filling NaNs…", flush=True)
    filled, mean_val, nan_mask = _fill_raster()
    vmin, vmax = np.nanmin(filled + mean_val), np.nanmax(filled + mean_val)

    sigmas       = [1, 4, 6, 10]
    chosen_sigma = DETREND_SIGMA_CHOSEN   # 6
    chosen_row   = sigmas.index(chosen_sigma)

    # Pre-compute all panels
    print("  Computing trends, residuals, FFTs…", flush=True)
    all_tr, all_res, all_fft_tr, all_fft_res = [], [], [], []
    for s in sigmas:
        tr, res = _gaussian_detrend(filled, s)
        all_tr.append(tr)
        all_res.append(res)
        all_fft_tr.append(_windowed_padded_fft(tr - np.mean(tr)))   # demean → no DC spike
        all_fft_res.append(_windowed_padded_fft(res))

    # Shared colour limits per column type
    res_abs     = max(np.max(np.abs(r)) for r in all_res)
    fft_tr_max  = max(f.max() for f in all_fft_tr)
    fft_res_max = max(f.max() for f in all_fft_res)

    col_titles = [
        "Trend surface",
        "FFT of trend\n(demeaned, windowed, padded)",
        "Residuals\n(filled − trend)",
        "FFT of residuals\n(windowed, padded)",
    ]
    col_labels = ["Depth (m)", "log |F|", "Residual (m)", "log |F|"]
    col_cmaps  = [CMAP_BATH, "hot", "RdBu_r", "hot"]
    col_ranges = [
        (vmin,     vmax),
        (0,        fft_tr_max),
        (-.5, .5),
        (0,        fft_res_max),
    ]

    n_rows, n_cols = len(sigmas), 4

    fig = plt.figure(figsize=(22, 14))
    gs  = fig.add_gridspec(n_rows, n_cols,
                           hspace=0.06, wspace=0.08,
                           left=0.08, right=0.97, top=0.91, bottom=0.11)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    for r, (s, tr, res, fft_tr, fft_res) in enumerate(
            zip(sigmas, all_tr, all_res, all_fft_tr, all_fft_res)):

        panels = [tr + mean_val, fft_tr, res, fft_res]

        for c, (data, (lo, hi), cmap) in enumerate(zip(panels, col_ranges, col_cmaps)):
            ax = axes[r][c]
            ax.imshow(data, cmap=cmap, origin="upper", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])

            # Column titles on the first row only
            if r == 0:
                ax.set_title(col_titles[c], fontsize=FS_TITLE, pad=7)

            # Row labels (sigma value) on the leftmost column
            if c == 0:
                ax.set_ylabel(
                    f"σ = {s} px\n({s * 20} m)" +
                    ("  ← chosen" if s == chosen_sigma else ""),
                    fontsize=FS, labelpad=8)

            # Orange border for the chosen sigma row
            if r == chosen_row:
                for sp in ax.spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR)
                    sp.set_linewidth(3)

    # Horizontal colorbars below each column.
    # Positions derived from gridspec: left=0.08, right=0.97, n_cols=4, wspace=0.08
    # col_width ≈ (0.97-0.08)/(4+3*0.08) ≈ 0.210;  gap ≈ 0.017
    col_x0 = [0.080, 0.307, 0.534, 0.761]
    col_w  = 0.210
    cbar_y, cbar_h = 0.042, 0.022

    for c in range(n_cols):
        cax = fig.add_axes([col_x0[c], cbar_y, col_w, cbar_h])
        ref_im = axes[0][c].images[0]
        cb = fig.colorbar(ref_im, cax=cax, orientation="horizontal")
        cb.set_label(col_labels[c], fontsize=FS - 1)
        cb.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Gaussian detrending sigma comparison  "
        f"(chosen: σ = {chosen_sigma} px = {chosen_sigma * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 8 — FFT preprocessing: windowing and zero-padding
# ---------------------------------------------------------------------------

def plot_fft_preprocessing(save_path=None):
    """
    Illustrate how Hann windowing and zero-padding reduce spectral leakage in
    the FFT of the detrended residuals.

    Top row (spatial domain):
        Residuals  |  Windowed residuals  |  2-D Hann window function
    Bottom row (frequency domain, log-scale):
        FFT – no preprocessing  |  FFT – windowed  |  FFT – windowed + padded
    Orange border marks the variant used in the pipeline.
    """
    filled, mean_val, nan_mask = _fill_raster()
    _, residuals = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)

    h, w = residuals.shape

    # 2-D Hann window (mirrors destripeClass._apply_window with type='hann')
    window   = np.outer(np.hanning(h), np.hanning(w))
    eps      = 1e-2                          # floor used in destripeClass
    win_clip = np.where(window < eps, eps, window)
    windowed = residuals * win_clip

    # Zero-padding (mirrors destripeClass._apply_padding)
    pad        = max(h, w) // 2
    win_padded = np.pad(windowed, pad, mode="constant")

    # FFT log-amplitude for each variant
    def _logfft(data):
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))) + 1)

    fft_raw = _logfft(residuals)
    fft_win = _logfft(windowed)
    fft_wp  = _logfft(win_padded)

    # Shared colour scale across all three FFT panels
    fft_max = max(fft_raw.max(), fft_win.max(), fft_wp.max())
    res_lim = 0.5

    # ---- layout: 2 rows × 3 cols ----
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.14, wspace=0.10,
                           left=0.04, right=0.97, top=0.87, bottom=0.05)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # ---- top row: spatial domain ----
    im0 = axes[0][0].imshow(residuals, cmap="RdBu_r", origin="upper",
                              vmin=-res_lim, vmax=res_lim)
    axes[0][0].set_title("Residuals\n(input to FFT)", fontsize=FS_TITLE, pad=7)
    fig.colorbar(im0, ax=axes[0][0], fraction=0.046, pad=0.03).set_label(
        "Residual (m)", fontsize=FS - 1)

    im1 = axes[0][1].imshow(windowed, cmap="RdBu_r", origin="upper",
                              vmin=-res_lim, vmax=res_lim)
    axes[0][1].set_title("Windowed residuals\n(Hann window applied)",
                          fontsize=FS_TITLE, pad=7)
    fig.colorbar(im1, ax=axes[0][1], fraction=0.046, pad=0.03).set_label(
        "Residual (m)", fontsize=FS - 1)

    im2 = axes[0][2].imshow(window, cmap="YlOrRd_r", origin="upper", vmin=0, vmax=1)
    axes[0][2].set_title("2-D Hann window\n(weights applied to residuals)",
                          fontsize=FS_TITLE, pad=7)
    cb2 = fig.colorbar(im2, ax=axes[0][2], fraction=0.046, pad=0.03)
    cb2.set_label("Weight  (0 → 1)", fontsize=FS - 1)
    cb2.set_ticks([0, 0.5, 1])

    # ---- bottom row: frequency domain ----
    axes[1][0].imshow(fft_raw, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
    axes[1][0].set_title(
        "FFT – no preprocessing\n(spectral leakage visible as diffuse glow)",
        fontsize=FS_TITLE, pad=7)

    axes[1][1].imshow(fft_win, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
    axes[1][1].set_title(
        "FFT – windowed, no padding\n(leakage reduced, stripe peak sharper)",
        fontsize=FS_TITLE, pad=7)

    im5 = axes[1][2].imshow(fft_wp, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
    axes[1][2].set_title(
        "FFT – windowed + zero-padded\n(used in pipeline; finest frequency resolution)",
        fontsize=FS_TITLE, pad=7)
    fig.colorbar(im5, ax=axes[1][2], fraction=0.046, pad=0.03).set_label(
        "log |F|", fontsize=FS - 1)

    # Orange border on the pipeline variant
    for sp in axes[1][2].spines.values():
        sp.set_edgecolor(HIGHLIGHT_COLOR)
        sp.set_linewidth(3)

    for row in axes:
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "FFT preprocessing: windowing and zero-padding reduce spectral leakage",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 9 — Notch filter width comparison
# ---------------------------------------------------------------------------

def plot_notch_width_comparison(save_path=None):
    """
    Four-row comparison of notch filter widths [1, 3, 5, 10].

    Each row shows:
      Col 0 : Windowed+padded FFT amplitude with the rotated notch band
               highlighted in orange — shows how wide a band is zeroed.
      Col 1 : Signal removed by the notch (residuals − filtered).
               At the right width this should contain only stripe artefacts.
      Col 2 : Filtered residuals (what remains after the notch).
      Col 3 : Bar chart — mean log-amplitude inside the notch band,
               normalised by the total mean (energy ratio).
               = sum(notch * log|F|) / sum(notch)

    Stripe angle is detected once from the windowed+padded FFT.
    Filtering uses the non-windowed padded residuals (mirrors pipeline).
    Orange border marks the chosen width = NOTCH_WIDTH_CHOSEN px.
    """
    print("  Filling NaNs…", flush=True)
    filled, mean_val, nan_mask = _fill_raster()
    _, residuals = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)

    h, w = residuals.shape

    # 2-D Hann window (mirrors destripeClass._apply_window, type='hann', eps=1e-2)
    window   = np.outer(np.hanning(h), np.hanning(w))
    eps      = 1e-2
    win_clip = np.where(window < eps, eps, window)
    windowed = residuals * win_clip

    # Padding (mirrors destripeClass._apply_padding: pad = max(h, w) // 2)
    pad            = max(h, w) // 2
    unpadded_slice = (slice(pad, pad + h), slice(pad, pad + w))

    # Windowed + padded FFT — used for angle detection, display, and metric
    win_padded = np.pad(windowed, pad, mode="constant")
    F_win      = np.fft.fftshift(np.fft.fft2(win_padded))
    log_amp    = np.log(np.abs(F_win) + 1)

    # Non-windowed + padded FFT — used for filtering (matches pipeline)
    nowin_padded = np.pad(residuals, pad, mode="constant")
    F_nowin      = np.fft.fftshift(np.fft.fft2(nowin_padded))

    # Detect stripe angle from the cleaner windowed FFT
    print("  Detecting stripe angle…", flush=True)
    angle, _, _ = _find_stripe_angle(log_amp)
    print(f"  Detected angle: {angle:.0f}°", flush=True)

    widths     = [1, 3, 5, 8]
    chosen_row = widths.index(NOTCH_WIDTH_CHOSEN)

    def _make_rotated_notch(shape, width):
        """Diagonal band of given half-width, rotated to stripe angle."""
        hp, wp = shape
        base   = np.zeros((hp, wp), dtype=float)
        for i in range(-width, width + 1):
            base += np.eye(hp, wp, k=i)
        cy, cx = hp // 2, wp // 2
        base[cy - NOTCH_CENTER_SIZE:cy + NOTCH_CENTER_SIZE,
             cx - NOTCH_CENTER_SIZE:cx + NOTCH_CENTER_SIZE] = 0
        base = np.clip(base, 0, 1)
        rot  = ndimage.rotate(base, angle=float(angle), reshape=False)
        return np.clip(rot, 0, 1)

    def _apply_notch(F, notch_mask):
        """Zero the notch band in F; return unpadded real-space result."""
        F_filt   = F * (1.0 - notch_mask)
        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt)))
        return filtered[unpadded_slice]

    # Pre-compute notch, filtered residuals, removed signal, and energy metric
    # Metric: mean log-amplitude inside the notch band (normalised by total mean)
    #   = sum(notch * log|F|) / sum(notch)
    # A value > global_mean indicates the band captures above-average energy
    global_mean_amp = float(np.mean(log_amp))
    print("  Filtering for each width…", flush=True)
    results = []
    for wid in widths:
        notch    = _make_rotated_notch(win_padded.shape, wid)
        filtered = _apply_notch(F_nowin, notch)
        removed  = residuals - filtered
        metric   = float(np.sum(notch * log_amp) / (np.sum(notch) + 1e-12))
        results.append((notch, filtered, removed, metric))

    # Colour limits — cols 1 and 2 are independent
    removed_max = max(float(np.max(np.abs(r[2]))) for r in results)
    res_lim     = float(np.max(np.abs(residuals)))
    fft_max     = float(log_amp.max())

    # ---- Layout: 4 rows × 3 cols ----
    n_rows, n_cols = len(widths), 3
    fig = plt.figure(figsize=(18, 15))
    gs  = fig.add_gridspec(n_rows, n_cols,
                           hspace=0.08, wspace=0.10,
                           left=0.12, right=0.97, top=0.92, bottom=0.08)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    col_titles = [
        f"FFT amplitude  +  notch band\n(stripe angle = {angle:.0f}°,  orange = zeroed region)",
        "Removed signal\n(residuals  −  filtered)",
        "Filtered residuals\n(stripe artefact removed)",
    ]

    for r, (wid, (notch, filtered, removed, metric)) in enumerate(zip(widths, results)):
        ax0, ax1, ax2 = axes[r]

        # Col 0: FFT amplitude with orange overlay for the zeroed band
        ax0.imshow(log_amp, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
        notch_show = np.ma.masked_where(notch < 0.3, notch)
        ax0.imshow(notch_show, cmap="Oranges", origin="upper", alpha=0.65, vmin=0, vmax=1)

        # Col 1: What the notch removed — own colour scale, metric annotated top-left
        ax1.imshow(removed, cmap="RdBu_r", origin="upper",
                   vmin=-removed_max, vmax=removed_max)
        ax1.text(0.03, 0.97, f"E = {metric:.3f}",
                 transform=ax1.transAxes, ha="left", va="top",
                 fontsize=FS - 2, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           alpha=0.75, edgecolor="none"))

        # Col 2: What the notch left behind — own colour scale (full residuals range)
        ax2.imshow(filtered, cmap="RdBu_r", origin="upper",
                   vmin=-res_lim, vmax=res_lim)

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([]); ax.set_yticks([])

        # Row label
        lbl = f"width = {wid} px\n({wid * 2 + 1} diagonals)"
        if wid == NOTCH_WIDTH_CHOSEN:
            lbl += "\n← chosen"
        ax0.set_ylabel(lbl, fontsize=FS, labelpad=8)

        # Orange border on chosen row
        if r == chosen_row:
            for c in range(n_cols):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR)
                    sp.set_linewidth(3)

        # Column titles on first row only
        if r == 0:
            for c, title in enumerate(col_titles):
                axes[0][c].set_title(title, fontsize=FS_TITLE, pad=7)

    # ---- Horizontal colorbars below each column ----
    fig.canvas.draw()   # force layout so get_position() is accurate
    cbar_y, cbar_h = 0.032, 0.020
    col_labels  = ["log |F|", "Removed (m)", "Residual (m)"]
    ref_images  = [axes[0][c].images[0] for c in range(n_cols)]

    for c in range(n_cols):
        pos = axes[-1][c].get_position()
        cax = fig.add_axes([pos.x0, cbar_y, pos.width, cbar_h])
        cb  = fig.colorbar(ref_images[c], cax=cax, orientation="horizontal")
        cb.set_label(col_labels[c], fontsize=FS - 1)
        cb.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Notch filter width comparison  "
        f"(chosen: width = {NOTCH_WIDTH_CHOSEN} px = {NOTCH_WIDTH_CHOSEN * 2 + 1} diagonals)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 10 — Low-pass / notch interaction: varying detrend sigma
# ---------------------------------------------------------------------------

def plot_lowpass_notch_interaction(save_path=None):
    """
    Four-row comparison showing how the Gaussian detrend sigma (low-pass cutoff)
    affects what the notch filter sees and removes.

    All rows use the same notch width (NOTCH_WIDTH_CHOSEN) and stripe angle
    (detected from the chosen sigma's FFT for consistency).  Only sigma varies.

    Columns:
      Col 0 : FFT of windowed+padded residuals with the rotated notch band
               overlaid in orange.
      Col 1 : Signal removed by the notch (residuals − filtered).
               Should be clean stripes at the chosen sigma; contains more
               real bathymetric signal at small sigma or large sigma.
      Col 2 : Filtered residuals (what the notch leaves behind).

    Orange border marks the chosen sigma = DETREND_SIGMA_CHOSEN.
    """
    print("  Filling NaNs…", flush=True)
    filled, mean_val, nan_mask = _fill_raster()

    sigmas      = [1, 4, 6, 10]
    chosen_row  = sigmas.index(DETREND_SIGMA_CHOSEN)

    # ---- Detect stripe angle from the chosen sigma (used for all rows) ----
    _, res_chosen = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)
    h, w = res_chosen.shape
    pad  = max(h, w) // 2
    unpadded_slice = (slice(pad, pad + h), slice(pad, pad + w))

    window   = np.outer(np.hanning(h), np.hanning(w))
    eps      = 1e-2
    win_clip = np.where(window < eps, eps, window)

    win_padded_chosen = np.pad(res_chosen * win_clip, pad, mode="constant")
    F_win_chosen      = np.fft.fftshift(np.fft.fft2(win_padded_chosen))
    log_amp_chosen    = np.log(np.abs(F_win_chosen) + 1)

    print("  Detecting stripe angle from chosen sigma…", flush=True)
    angle, _, _ = _find_stripe_angle(log_amp_chosen)
    print(f"  Detected angle: {angle:.0f}°", flush=True)

    # ---- Build fixed notch (width=NOTCH_WIDTH_CHOSEN, rotated to stripe angle) ----
    padded_shape = win_padded_chosen.shape

    def _make_notch(shape):
        hp, wp = shape
        base   = np.zeros((hp, wp), dtype=float)
        for i in range(-NOTCH_WIDTH_CHOSEN, NOTCH_WIDTH_CHOSEN + 1):
            base += np.eye(hp, wp, k=i)
        cy, cx = hp // 2, wp // 2
        base[cy - NOTCH_CENTER_SIZE:cy + NOTCH_CENTER_SIZE,
             cx - NOTCH_CENTER_SIZE:cx + NOTCH_CENTER_SIZE] = 0
        base = np.clip(base, 0, 1)
        rot  = ndimage.rotate(base, angle=float(angle), reshape=False)
        return np.clip(rot, 0, 1)

    notch = _make_notch(padded_shape)

    def _apply_notch(F_nowin):
        F_filt   = F_nowin * (1.0 - notch)
        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt)))
        return filtered[unpadded_slice]

    # ---- Pre-compute for all sigmas ----
    print("  Computing residuals, FFTs, and filtering for each sigma…", flush=True)
    all_log_amp, all_filtered, all_removed = [], [], []

    for s in sigmas:
        _, res = _gaussian_detrend(filled, s)

        # Windowed+padded FFT — for display
        win_pad = np.pad(res * win_clip, pad, mode="constant")
        F_win   = np.fft.fftshift(np.fft.fft2(win_pad))
        log_amp = np.log(np.abs(F_win) + 1)
        all_log_amp.append(log_amp)

        # Non-windowed+padded FFT — for filtering (matches pipeline)
        nowin_pad = np.pad(res, pad, mode="constant")
        F_nowin   = np.fft.fftshift(np.fft.fft2(nowin_pad))
        filtered  = _apply_notch(F_nowin)
        all_filtered.append(filtered)
        all_removed.append(res - filtered)

    # ---- Shared colour limits ----
    # Each sigma produces differently-scaled residuals; use per-row residual range
    # for cols 1+2 so comparisons are visible within each row.
    # For col 0 (FFT) use a common scale so brightness is comparable across rows.
    fft_max = max(la.max() for la in all_log_amp)

    # ---- Layout: 4 rows × 3 cols ----
    n_rows, n_cols = len(sigmas), 3
    fig = plt.figure(figsize=(18, 15))
    gs  = fig.add_gridspec(n_rows, n_cols,
                           hspace=0.08, wspace=0.10,
                           left=0.12, right=0.97, top=0.92, bottom=0.08)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    col_titles = [
        f"FFT of residuals  +  notch band\n(angle = {angle:.0f}°,  width = {NOTCH_WIDTH_CHOSEN} px,  orange = zeroed)",
        "Removed signal\n(residuals  −  filtered)",
        "Filtered residuals\n(stripe artefact removed)",
    ]

    last_im_col1 = None   # for colorbar reference

    for r, (s, log_amp, filtered, removed) in enumerate(
            zip(sigmas, all_log_amp, all_filtered, all_removed)):

        ax0, ax1, ax2 = axes[r]

        # Per-row colour limit for spatial panels (residuals differ a lot between sigmas)
        res   = all_removed[r] + filtered          # recover original residuals
        sym   = float(np.max(np.abs(res)))

        # Col 0: FFT + orange notch overlay (shared scale across rows)
        ax0.imshow(log_amp, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
        notch_show = np.ma.masked_where(notch < 0.3, notch)
        ax0.imshow(notch_show, cmap="Oranges", origin="upper", alpha=0.65, vmin=0, vmax=1)

        # Col 1: Removed signal (per-row scale so pattern is visible at all sigmas)
        sym_rm = float(np.max(np.abs(removed)))
        im1 = ax1.imshow(removed, cmap="RdBu_r", origin="upper",
                         vmin=-sym_rm, vmax=sym_rm)

        # Col 2: Filtered residuals (same per-row scale as col 1 for direct comparison)
        im2 = ax2.imshow(filtered, cmap="RdBu_r", origin="upper",
                         vmin=-sym_rm, vmax=sym_rm)

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([]); ax.set_yticks([])

        # Inline colorbars on cols 1+2 so each row has its own scale label
        for ax, im in [(ax1, im1), (ax2, im2)]:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                              orientation="vertical")
            cb.set_label("m", fontsize=FS - 3)
            cb.ax.tick_params(labelsize=FS - 4)

        # Row label (sigma value)
        lbl = f"σ = {s} px  ({s * 20} m)"
        if s == DETREND_SIGMA_CHOSEN:
            lbl += "\n← chosen"
        ax0.set_ylabel(lbl, fontsize=FS, labelpad=8)

        # Orange border on chosen row
        if r == chosen_row:
            for c in range(n_cols):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR)
                    sp.set_linewidth(3)

        # Column titles on first row only
        if r == 0:
            for c, title in enumerate(col_titles):
                axes[0][c].set_title(title, fontsize=FS_TITLE, pad=7)

    # ---- Single colorbar for col 0 (FFT, shared scale) ----
    col0_right = 0.120 + 0.263   # right edge of col-0 axes ≈ 0.383
    cax0 = fig.add_axes([0.120, 0.032, 0.263, 0.020])
    cb0  = fig.colorbar(axes[0][0].images[0], cax=cax0, orientation="horizontal")
    cb0.set_label("log |F|", fontsize=FS - 1)
    cb0.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Low-pass / notch interaction: effect of detrend sigma  "
        f"(notch width fixed at {NOTCH_WIDTH_CHOSEN} px,  chosen σ = {DETREND_SIGMA_CHOSEN} px = {DETREND_SIGMA_CHOSEN * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 11 — Stripe angle detection: two examples
# ---------------------------------------------------------------------------

def plot_angle_detection(save_path=None):
    """
    Two-row comparison of stripe-angle detection on two contrasting examples.

    Row 0 — cell_10_CDI_3175974: clean stripe noise → easy detection.
    Row 1 — cell_12_CDI_3466114: dominant sand-wave energy → the absolute
             response alone is noisy; the relative response (vs. neighbouring
             angles) correctly isolates the stripe peak.

    For each example four panels are shown left-to-right:
      Col 0 : Windowed + zero-padded FFT log amplitude.
      Col 1 : Same FFT with the detected-angle notch band overlaid in orange,
               showing which frequency region the filter zeros out.
      Col 2 : Angle sweep — absolute mean log-amplitude inside the notch band
               as a function of rotation angle (0–180°, step 2°).
      Col 3 : Angle sweep — relative response: absolute minus the mean of the
               eight neighbouring-angle responses (±~8° window), which removes
               the isotropic sand-wave "floor" and sharpens the stripe peak.

    Detected angle is marked with a vertical dashed orange line in cols 2+3.
    """
    examples = [
        (RASTER_PATH,
         "cell_10  ·  CDI 3175974\n(clear stripe noise — easy case)"),
        (RASTER_PATH_2,
         "cell_12  ·  CDI 3466114\n(high sand-wave energy — harder)"),
    ]

    print("  Processing examples…", flush=True)
    all_data = []
    for path, label in examples:
        print(f"    Filling NaNs: {path}", flush=True)
        filled, mean_val, nan_mask = _fill_raster_path(path)
        _, residuals = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)

        h, w     = residuals.shape
        window   = np.outer(np.hanning(h), np.hanning(w))
        eps      = 1e-2
        win_clip = np.where(window < eps, eps, window)
        pad      = max(h, w) // 2

        win_padded = np.pad(residuals * win_clip, pad, mode="constant")
        F_win      = np.fft.fftshift(np.fft.fft2(win_padded))
        log_amp    = np.log(np.abs(F_win) + 1)

        print("    Angle sweep…", flush=True)
        angle, resp, rel = _find_stripe_angle(log_amp)
        print(f"    Detected angle: {angle:.0f}°", flush=True)

        # Build notch rotated to detected angle (matches pipeline)
        hp, wp = win_padded.shape
        base   = np.zeros((hp, wp), dtype=float)
        for i in range(-NOTCH_WIDTH_CHOSEN, NOTCH_WIDTH_CHOSEN + 1):
            base += np.eye(hp, wp, k=i)
        cy, cx = hp // 2, wp // 2
        base[cy - NOTCH_CENTER_SIZE:cy + NOTCH_CENTER_SIZE,
             cx - NOTCH_CENTER_SIZE:cx + NOTCH_CENTER_SIZE] = 0
        base  = np.clip(base, 0, 1)
        notch = ndimage.rotate(base, float(angle), reshape=False)
        notch = np.clip(notch, 0, 1)

        all_data.append(dict(
            log_amp=log_amp,
            notch=notch,
            angles=np.arange(0, 180, 1),   # step=2 matches _find_stripe_angle
            resp=resp,
            rel=rel,
            angle=angle,
            label=label,
        ))

    # ---- Layout: 2 rows × 4 cols ----------------------------------------
    # Image columns are square; 1D-plot columns get 50% extra width.
    fig = plt.figure(figsize=(23, 12))
    gs  = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1.5, 1.5],
        hspace=0.22, wspace=0.25,
        left=0.09, right=0.98, top=0.90, bottom=0.09,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    col_titles = [
        "FFT log amplitude\n(windowed + zero-padded)",
        f"FFT + notch band\n(orange = filtered region,  width = {NOTCH_WIDTH_CHOSEN} px)",
        "Absolute response vs. angle\n(mean log |F| inside notch band)",
        "Relative response vs. angle\n(absolute − local-neighbour mean,  ±8° window)",
    ]

    for r, d in enumerate(all_data):
        ax0, ax1, ax2, ax3 = axes[r]
        fft_max = float(d["log_amp"].max())

        # ---- Col 0: raw FFT ------------------------------------------------
        ax0.imshow(d["log_amp"], cmap="hot", origin="upper", vmin=0, vmax=fft_max)
        cb0 = fig.colorbar(ax0.images[0], ax=ax0, fraction=0.046, pad=0.02)
        cb0.set_label("log |F|", fontsize=FS - 2)
        cb0.ax.tick_params(labelsize=FS - 4)

        # ---- Col 1: FFT + orange notch overlay -----------------------------
        ax1.imshow(d["log_amp"], cmap="hot", origin="upper", vmin=0, vmax=fft_max)
        notch_show = np.ma.masked_where(d["notch"] < 0.3, d["notch"])
        ax1.imshow(notch_show, cmap="Oranges", origin="upper", alpha=0.65,
                   vmin=0, vmax=1)
        ax1.text(0.03, 0.97, f"θ = {d['angle']:.0f}°",
                 transform=ax1.transAxes, ha="left", va="top",
                 fontsize=FS - 1, fontweight="bold", color=HIGHLIGHT_COLOR,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                           alpha=0.55, edgecolor="none"))

        for ax in [ax0, ax1]:
            ax.set_xticks([]); ax.set_yticks([])

        # ---- Col 2: Absolute response --------------------------------------
        ax2.plot(d["angles"], d["resp"], color="#1f77b4", lw=1.8,
                 label="Mean log |F|")
        ax2.axvline(d["angle"], color=HIGHLIGHT_COLOR, lw=2.0, ls="--",
                    label=f"Detected: {d['angle']:.0f}°")
        ax2.set_xlabel("Angle (°)", fontsize=FS - 1)
        ax2.set_ylabel("Mean log |F| in band", fontsize=FS - 1)
        ax2.set_xlim(0, 178)
        ax2.legend(fontsize=FS - 3, loc="best")
        ax2.grid(alpha=0.3)
        ax2.tick_params(labelsize=FS - 3)

        # ---- Col 3: Relative response --------------------------------------
        ax3.plot(d["angles"], d["rel"], color="#2ca02c", lw=1.8,
                 label="Relative response")
        ax3.axvline(d["angle"], color=HIGHLIGHT_COLOR, lw=2.0, ls="--",
                    label=f"Detected: {d['angle']:.0f}°")
        ax3.axhline(0.0, color="gray", lw=0.9, ls=":", zorder=0)
        ax3.set_xlabel("Angle (°)", fontsize=FS - 1)
        ax3.set_ylabel("Response − local mean", fontsize=FS - 1)
        ax3.set_xlim(0, 178)
        ax3.legend(fontsize=FS - 3, loc="best")
        ax3.grid(alpha=0.3)
        ax3.tick_params(labelsize=FS - 3)

        # Row label on leftmost axis
        ax0.set_ylabel(d["label"], fontsize=FS - 1, labelpad=10)

        # Column titles on first row only
        if r == 0:
            for c, title in enumerate(col_titles):
                axes[0][c].set_title(title, fontsize=FS_TITLE, pad=7)

    fig.suptitle(
        "Stripe angle detection: FFT-based angular scan (0–180°,  step 2°)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97,
    )

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # plot_extrapolation_stages   (save_path=os.path.join(SAVE_DIR, "01_stages.png"))
    # plot_blend_factor           (save_path=os.path.join(SAVE_DIR, "02_blend_factor.png"))
    # plot_degree_comparison      (save_path=os.path.join(SAVE_DIR, "03_degree_comparison.png"))
    # plot_smooth_comparison      (save_path=os.path.join(SAVE_DIR, "04_smooth_comparison.png"))
    # plot_blend_dist_comparison  (save_path=os.path.join(SAVE_DIR, "05_blend_dist_comparison.png"))
    # plot_detrend_stages          (save_path=os.path.join(SAVE_DIR, "06_detrend_stages.png"))
    # plot_detrend_sigma_comparison(save_path=os.path.join(SAVE_DIR, "07_detrend_sigma_comparison.png"))
    # plot_fft_preprocessing        (save_path=os.path.join(SAVE_DIR, "08_fft_preprocessing.png"))
    # plot_notch_width_comparison   (save_path=os.path.join(SAVE_DIR, "09_notch_width_comparison.png"))
    # plot_lowpass_notch_interaction(save_path=os.path.join(SAVE_DIR, "10_lowpass_notch_interaction.png"))
    plot_angle_detection          (save_path=os.path.join(SAVE_DIR, "11_angle_detection.png"))
    print(f"\nAll plots saved to: {SAVE_DIR}/")
