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
# Sandwave-detection constants
# ---------------------------------------------------------------------------

SWD_CELL        = 47
SWD_CDI         = 3844669
SWD_RASTER_PATH = (f"destriped_rasters/cell_{SWD_CELL}_CDI_{SWD_CDI}_destriped.npy")
SWD_LABELS_PATH = (f"sandwave_detection_v8/labels/"
                   f"cell_{SWD_CELL}_CDI_{SWD_CDI}_destriped_labels_smoothed.npy")

SWD_DETREND_SIGMA = 30   # pixels  (= 600 m at 20 m/px)
SWD_STD_SIZE      = 10   # pixels  (= 200 m)
SWD_GRAD_SMOOTH   = 5    # pixels  (= 100 m)

CMAP_FEAT        = "inferno"
SW_CONTOUR_COLOR = "#00e676"   # bright green for the sandwave boundary

# Label-cleaning hyperparameters (from detect_sws.clean_smoothed_labels)
SWD_CLEAN_SIGMA   = 20    # px  Gaussian smoothing sigma
SWD_CLEAN_THRESH  = 0.35  # binary threshold on smoothed probability
SWD_CLOSING_ITER  = 2     # binary closing iterations (3×3 structure)
SWD_MIN_PIXELS    = 200   # minimum connected component size (px)
SWD_RAW_LABELS_PATH = (f"sandwave_detection_v8/labels/"
                        f"cell_{SWD_CELL}_CDI_{SWD_CDI}_destriped_labels.npy")

# Erosion-effect example (cell 14 – thin features make erosion visible)
SWD_EROSION_CELL = 14
SWD_EROSION_CDI  = 3844669
SWD_EROSION_RASTER_PATH = (f"destriped_rasters/"
                            f"cell_{SWD_EROSION_CELL}_CDI_{SWD_EROSION_CDI}_destriped.npy")
SWD_EROSION_RAW_PATH    = (f"sandwave_detection_v8/labels/"
                            f"cell_{SWD_EROSION_CELL}_CDI_{SWD_EROSION_CDI}"
                            f"_destriped_labels.npy")
SWD_EROSION_SMOOTH_PATH = (f"sandwave_detection_v8/labels/"
                            f"cell_{SWD_EROSION_CELL}_CDI_{SWD_EROSION_CDI}"
                            f"_destriped_labels_smoothed.npy")


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

    best_row = raster_dm.shape[0] // 2

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
# Sandwave-detection helpers
# ---------------------------------------------------------------------------

def _load_swd():
    """Load, demean, and NaN-fill the sandwave detection example raster."""
    return _fill_raster_path(SWD_RASTER_PATH)


def _load_swd_labels():
    """Load smoothed binary labels (−1=no-data, 0=flat seabed, 1=sandwave)."""
    return np.load(SWD_LABELS_PATH)


def _local_std(data, size):
    """Local standard deviation in a size × size sliding window.

    Uses Var[x] = E[x²] − E[x]² computed via uniform_filter → O(N).
    """
    E_x2 = ndimage.uniform_filter(data ** 2, size=size)
    E_x  = ndimage.uniform_filter(data,      size=size)
    return np.sqrt(np.maximum(E_x2 - E_x ** 2, 0.0))


def _local_grad(data, smooth):
    """Gradient magnitude |∇f| after Gaussian pre-smoothing (σ = smooth pixels)."""
    smoothed = ndimage.gaussian_filter(data, sigma=smooth)
    gy, gx   = np.gradient(smoothed)
    return np.sqrt(gx ** 2 + gy ** 2)


def _sw_contour(ax, labels, lw=1.5):
    """Overlay the sandwave-area boundary on *ax* as a contour line."""
    mask = (labels == 1).astype(float)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[SW_CONTOUR_COLOR],
                   linewidths=lw, alpha=0.9)


def _labels_cmap_norm():
    """3-class colormap/norm for label images (−1→gray, 0→white, 1→red)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#cccccc", "white", "#e53935"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    return cmap, norm


def _load_swd_raw_labels():
    """Load raw (un-cleaned) K-Means labels (−1=no-data, 0=flat, 1=sandwave)."""
    return np.load(SWD_RAW_LABELS_PATH)


def _clean_steps(labels_for_clean, valid_mask, sigma, threshold, closing_iter, min_pixels):
    """
    Re-implement clean_smoothed_labels step-by-step and return intermediates.

    Parameters
    ----------
    labels_for_clean : int ndarray  (0 or 1 only — no −1 values)
    valid_mask       : bool ndarray (True where bathymetry data is valid)

    Returns a dict with keys:
      smoothed_float  — continuous probability 0–1 (NaN outside valid area)
      after_threshold — label image after thresholding
      after_closing   — after binary closing
      after_fillholes — after fill_holes
      final           — after removing components smaller than min_pixels
    Label images use −1=no-data, 0=flat, 1=sandwave convention.
    """
    labels_f = labels_for_clean.astype(np.float32)
    valid_f  = valid_mask.astype(np.float32)

    # Step 1: support-normalised Gaussian smoothing
    weighted = ndimage.gaussian_filter(labels_f * valid_f, sigma=sigma, mode="nearest")
    support  = ndimage.gaussian_filter(valid_f,            sigma=sigma, mode="nearest")
    smooth_f = np.divide(weighted, support,
                         out=np.zeros_like(weighted, dtype=np.float32),
                         where=support > 1e-6)

    # Step 2: threshold + restrict to valid pixels
    binary = (smooth_f > threshold) & valid_mask

    # Step 3: binary closing (bridges small gaps)
    closed = ndimage.binary_closing(
        binary,
        structure=np.ones((3, 3), dtype=bool),
        iterations=closing_iter,
        border_value=1,
    )

    # Step 4: fill internal holes
    filled = ndimage.binary_fill_holes(closed)

    # Step 5: remove components smaller than min_pixels
    lab_comp, n_comp = ndimage.label(filled)
    if n_comp > 0:
        sizes     = np.bincount(lab_comp.ravel())
        keep      = sizes >= min_pixels
        keep[0]   = False
        final_bin = keep[lab_comp]
    else:
        final_bin = filled.copy()

    def _to_label(mask):
        out = np.zeros(labels_for_clean.shape, dtype=np.int8)
        out[mask]         = 1
        out[~valid_mask]  = -1
        return out

    return {
        "smoothed_float":  np.where(valid_mask, smooth_f.astype(float), np.nan),
        "after_threshold": _to_label(binary),
        "after_closing":   _to_label(closed),
        "after_fillholes": _to_label(filled),
        "final":           _to_label(final_bin),
    }


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
# Figure 12 — Why filtering uses the unwindowed FFT (invertibility argument)
# ---------------------------------------------------------------------------

def plot_windowed_vs_unwindowed_filtering(save_path=None):
    """
    Two-row diagram showing why the notch filter is applied to the
    *unwindowed* FFT rather than the windowed one.

    Windowing (Hann) is useful for angle detection: it suppresses spectral
    leakage so stripe peaks stand out clearly.  But filtering the windowed
    FFT and inverting does NOT give the filtered residuals — it gives a
    windowed version of them.  To recover the true filtered residuals one
    would divide by the window, which amplifies the near-zero edge weights
    (eps-clipped to 0.01) by up to 100×, producing catastrophic artefacts.

    The pipeline avoids this by keeping two parallel FFT paths:
      • Windowed FFT  → angle detection only.
      • Unwindowed FFT → notch filtering; iFFT is directly the filtered
        residuals with no division or correction required.

    Layout: 2 rows × 4 columns
      Row 0  (windowed path — not used for filtering):
        Col 0  Residuals × Hann window (spatial input, edges ≈ 0).
        Col 1  FFT log|F|  (clean spectrum; used for angle detection).
        Col 2  iFFT after notch (output is windowed → edges still ≈ 0).
        Col 3  ÷ Hann window to recover residuals → edge artefacts explode.
      Row 1  (unwindowed path — pipeline):
        Col 0  Residuals (full amplitude, unmodified).
        Col 1  FFT log|F|  (some spectral leakage, acceptable for filtering).
        Col 2  iFFT after notch (correctly filtered residuals, clean edges).
        Col 3  Removed stripes = residuals − filtered (clean stripe shape).

    Orange border = correct path.  Red border = problematic panel.
    """
    print("  Filling NaNs…", flush=True)
    filled, mean_val, nan_mask = _fill_raster()
    _, residuals = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)

    h, w = residuals.shape

    # ---- 2-D Hann window, eps-clipped (mirrors destripeClass._apply_window) ----
    window   = np.outer(np.hanning(h), np.hanning(w))
    eps      = 1e-2
    win_clip = np.where(window < eps, eps, window)
    windowed = residuals * win_clip

    # ---- Zero-padding (mirrors destripeClass._apply_padding) ----
    pad            = max(h, w) // 2
    unpadded_slice = (slice(pad, pad + h), slice(pad, pad + w))
    win_padded     = np.pad(windowed,  pad, mode="constant")
    nowin_padded   = np.pad(residuals, pad, mode="constant")

    # ---- FFTs ----
    F_win   = np.fft.fftshift(np.fft.fft2(win_padded))
    F_nowin = np.fft.fftshift(np.fft.fft2(nowin_padded))
    log_win   = np.log(np.abs(F_win)   + 1)
    log_nowin = np.log(np.abs(F_nowin) + 1)

    # ---- Detect stripe angle from windowed FFT ----
    print("  Detecting stripe angle…", flush=True)
    angle, _, _ = _find_stripe_angle(log_win)
    print(f"  Detected angle: {angle:.0f}°", flush=True)

    # ---- Build rotated notch (padded-array shape) ----
    hp, wp = win_padded.shape
    base   = np.zeros((hp, wp), dtype=float)
    for i in range(-NOTCH_WIDTH_CHOSEN, NOTCH_WIDTH_CHOSEN + 1):
        base += np.eye(hp, wp, k=i)
    cy, cx = hp // 2, wp // 2
    base[cy - NOTCH_CENTER_SIZE:cy + NOTCH_CENTER_SIZE,
         cx - NOTCH_CENTER_SIZE:cx + NOTCH_CENTER_SIZE] = 0
    notch = np.clip(
        ndimage.rotate(np.clip(base, 0, 1), float(angle), reshape=False),
        0, 1,
    )

    # ---- Windowed path ----
    ifft_win_crop = np.real(
        np.fft.ifft2(np.fft.ifftshift(F_win * (1.0 - notch)))
    )[unpadded_slice]                           # windowed filtered residuals
    recovered_win = ifft_win_crop / win_clip    # ÷ window → edge blow-up

    # ---- Unwindowed path (pipeline) ----
    filtered_nowin = np.real(
        np.fft.ifft2(np.fft.ifftshift(F_nowin * (1.0 - notch)))
    )[unpadded_slice]                           # correctly filtered residuals
    removed_nowin = residuals - filtered_nowin  # actual stripes extracted

    # ---- Colour limits ----
    res_lim      = float(np.max(np.abs(residuals)))
    removed_lim  = float(np.max(np.abs(removed_nowin)))
    blowup_lim   = float(np.max(np.abs(recovered_win)))   # will be >> res_lim
    fft_max      = max(log_win.max(), log_nowin.max())

    # ---- Layout: 2 rows × 4 cols ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.subplots_adjust(hspace=0.13, wspace=0.06,
                        left=0.10, right=0.97, top=0.88, bottom=0.09)

    col_titles = [
        "Spatial input",
        f"FFT  log|F|  (θ = {angle:.0f}°)",
        "iFFT  after notch",
        "Final step",
    ]
    row_labels = [
        "Windowed path\n(not used for filtering)",
        "Unwindowed path\n(pipeline)",
    ]

    # ---- Row 0: windowed path ----
    axes[0][0].imshow(windowed,      cmap="RdBu_r", origin="upper",
                      vmin=-res_lim,    vmax=res_lim)
    axes[0][1].imshow(log_win,       cmap="hot",    origin="upper",
                      vmin=0,           vmax=fft_max)
    axes[0][2].imshow(ifft_win_crop, cmap="RdBu_r", origin="upper",
                      vmin=-res_lim,    vmax=res_lim)
    axes[0][3].imshow(recovered_win, cmap="RdBu_r", origin="upper",
                      vmin=-blowup_lim, vmax=blowup_lim)

    # ---- Row 1: unwindowed path (pipeline) ----
    axes[1][0].imshow(residuals,      cmap="RdBu_r", origin="upper",
                      vmin=-res_lim,   vmax=res_lim)
    axes[1][1].imshow(log_nowin,      cmap="hot",    origin="upper",
                      vmin=0,          vmax=fft_max)
    axes[1][2].imshow(filtered_nowin, cmap="RdBu_r", origin="upper",
                      vmin=-res_lim,   vmax=res_lim)
    axes[1][3].imshow(removed_nowin,  cmap="RdBu_r", origin="upper",
                      vmin=-removed_lim, vmax=removed_lim)

    # ---- Annotations ----
    def _ann(ax, text, color="white", bg="#333333", fw="normal"):
        ax.text(0.03, 0.97, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=FS - 3, color=color,
                fontweight=fw,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=bg,
                          alpha=0.80, edgecolor="none"))

    _ann(axes[0][0], "edges → 0\n(Hann window)")
    _ann(axes[0][2], "output windowed\n(edges still ≈ 0)")
    _ann(axes[0][3],
         f"edge artefacts\n(max {blowup_lim:.2f} m  ≈  {blowup_lim/res_lim:.0f}× residual range)",
         color="#cc0000", bg="white", fw="bold")
    _ann(axes[1][1], "spectral leakage\n(acceptable for filtering)")
    _ann(axes[1][3], f"max removal: {removed_lim:.2f} m\n(stripe signal only)")

    # ---- Titles, row labels, tick removal ----
    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=FS_TITLE, pad=7)
    for r, lbl in enumerate(row_labels):
        axes[r][0].set_ylabel(lbl, fontsize=FS, labelpad=8)
    for r in range(2):
        for c in range(4):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

    # ---- Spine highlighting ----
    for c in range(4):                                    # orange = correct row
        for sp in axes[1][c].spines.values():
            sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(2.5)
    for sp in axes[0][3].spines.values():                # red = problem panel
        sp.set_edgecolor("#cc0000"); sp.set_linewidth(2.5)

    # ---- Arrows between columns ----
    for r in range(2):
        for c in range(3):
            axes[r][c].annotate(
                "", xy=(1.04, 0.5), xycoords="axes fraction",
                xytext=(1.0,  0.5), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # ---- Colorbars (one per column, below bottom row) ----
    fig.canvas.draw()
    cbar_y, cbar_h = 0.025, 0.018
    # Col 3 has different scales per row — use row 1 (correct) for the bar label
    col3_ref_img = axes[1][3].images[0]
    ref_imgs  = [axes[0][0].images[0], axes[0][1].images[0],
                 axes[0][2].images[0], col3_ref_img]
    col_units = ["Residual (m)", "log |F|", "Residual (m)", "Signal (m)"]
    for c in range(4):
        pos = axes[1][c].get_position()
        cax = fig.add_axes([pos.x0, cbar_y, pos.width, cbar_h])
        cb  = fig.colorbar(ref_imgs[c], cax=cax, orientation="horizontal")
        cb.set_label(col_units[c], fontsize=FS - 2)
        cb.ax.tick_params(labelsize=FS - 4)

    fig.suptitle(
        "Why the notch filter is applied to the unwindowed FFT",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.96)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 13 — Sandwave detection: residuals (detrending sigma comparison)
# ---------------------------------------------------------------------------

def plot_swd_residuals(save_path=None):
    """
    Justify the Gaussian detrending sigma = 30 px (600 m) for residual extraction.

    Layout: 2 rows × 5 columns (σ = 5, 15, 30, 60, 100 px)
      Row 0  Gaussian trend surface.
      Row 1  Residual = filled − trend, with sandwave boundary (green contour).

    Small σ → trend too localised → residual noisy, sandwave signal diluted.
    Large σ → trend over-smooth  → residual retains large-scale bathymetric slope.
    Chosen σ = 30 px (600 m) is highlighted with an orange border.
    """
    from matplotlib.lines import Line2D

    print("  Loading sandwave raster …", flush=True)
    filled, mean_val, nan_mask = _load_swd()
    labels = _load_swd_labels()

    sigmas     = [5, 15, 30, 60, 100]
    chosen_idx = sigmas.index(SWD_DETREND_SIGMA)

    all_tr, all_res = [], []
    for s in sigmas:
        tr, res = _gaussian_detrend(filled, s)
        all_tr.append(tr)
        all_res.append(res)

    vmin    = float(np.nanmin(filled + mean_val))
    vmax    = float(np.nanmax(filled + mean_val))
    res_abs = float(max(np.max(np.abs(r)) for r in all_res))

    cm_bath = copy.copy(CMAP_BATH);   cm_bath.set_bad(NAN_COLOR)
    cm_rdbu = copy.copy(plt.cm.RdBu_r); cm_rdbu.set_bad(NAN_COLOR)

    n_rows, n_cols = 2, len(sigmas)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9))
    fig.subplots_adjust(hspace=0.08, wspace=0.06,
                        left=0.09, right=0.97, top=0.88, bottom=0.13)

    im_tr_ref = im_res_ref = None
    for c, (s, tr, res) in enumerate(zip(sigmas, all_tr, all_res)):
        ax0, ax1 = axes[0][c], axes[1][c]

        # Row 0 — trend
        im_tr = ax0.imshow(
            np.ma.masked_where(nan_mask, tr + mean_val),
            cmap=cm_bath, origin="upper", vmin=vmin, vmax=vmax)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f"σ = {s} px  ({s * 20} m)", fontsize=FS_TITLE, pad=6)
        if c == 0:
            ax0.set_ylabel("Gaussian trend", fontsize=FS, labelpad=8)

        # Row 1 — residual + labels contour
        im_res = ax1.imshow(
            np.ma.masked_where(nan_mask, res),
            cmap=cm_rdbu, origin="upper", vmin=-res_abs, vmax=res_abs)
        ax1.set_xticks([]); ax1.set_yticks([])
        _sw_contour(ax1, labels)
        if c == 0:
            ax1.set_ylabel("Residual = filled − trend", fontsize=FS, labelpad=8)

        # Orange border for chosen column
        if c == chosen_idx:
            im_tr_ref = im_tr
            im_res_ref = im_res
            for r in range(n_rows):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    # Green contour legend
    leg = [Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                  label="sandwave area (smoothed labels)")]
    axes[1][-1].legend(handles=leg, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    # Colorbars — two side-by-side spanning full plot width
    fig.canvas.draw()
    cbar_y, cbar_h = 0.040, 0.022
    pos_tl = axes[0][0].get_position()
    pos_tr = axes[0][-1].get_position()
    full_w = pos_tr.x1 - pos_tl.x0
    half_w = full_w * 0.47

    cax0 = fig.add_axes([pos_tl.x0,              cbar_y, half_w, cbar_h])
    cax1 = fig.add_axes([pos_tl.x0 + full_w * 0.53, cbar_y, half_w, cbar_h])

    cb0 = fig.colorbar(im_tr_ref,  cax=cax0, orientation="horizontal")
    cb0.set_label("Depth (m)",     fontsize=FS - 1)
    cb0.ax.tick_params(labelsize=FS - 3)

    cb1 = fig.colorbar(im_res_ref, cax=cax1, orientation="horizontal")
    cb1.set_label("Residual (m)",  fontsize=FS - 1)
    cb1.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Detrending sigma — residual feature  "
        f"(chosen: σ = {SWD_DETREND_SIGMA} px = {SWD_DETREND_SIGMA * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.96)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 14 — Sandwave detection: local standard deviation (window-size comparison)
# ---------------------------------------------------------------------------

def plot_swd_local_std(save_path=None):
    """
    Justify the local-std window size = 10 px (200 m) for sandwave feature extraction.

    Layout: 1 row × 6 columns (window sizes 3, 5, 10, 20, 40 px + labels reference)
      Each panel shows the local σ(z) map with the sandwave boundary as a green
      contour.  The last column shows the smoothed classification labels.

    Small window → local std noisy, reflects instrument noise more than bedform relief.
    Large window → local std over-smoothed, loses spatial resolution of sandwave fields.
    Chosen size = 10 px (200 m) highlighted with orange border.
    """
    from matplotlib.lines import Line2D

    print("  Loading sandwave raster …", flush=True)
    filled, mean_val, nan_mask = _load_swd()
    labels = _load_swd_labels()
    cmap_lab, norm_lab = _labels_cmap_norm()

    # Compute residuals first (local std is applied to residuals, consistent with detect_sws)
    _, residuals = _gaussian_detrend(filled, SWD_DETREND_SIGMA)

    sizes      = [3, 5, 10, 20, 40]
    chosen_idx = sizes.index(SWD_STD_SIZE)

    all_std = []
    for sz in sizes:
        s = _local_std(residuals, sz)
        s[nan_mask] = np.nan
        all_std.append(s)

    feat_max = float(max(np.nanmax(s) for s in all_std))

    cm_feat = copy.copy(plt.colormaps[CMAP_FEAT]); cm_feat.set_bad(NAN_COLOR)

    n_cols = len(sizes) + 1   # last col = labels
    fig, axes = plt.subplots(1, n_cols, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.06, left=0.03, right=0.97, top=0.84, bottom=0.17)

    im_feat_ref = None
    for c, (sz, std_map) in enumerate(zip(sizes, all_std)):
        ax = axes[c]
        im = ax.imshow(
            np.ma.masked_where(nan_mask, std_map),
            cmap=cm_feat, origin="upper", vmin=0, vmax=feat_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"size = {sz} px\n({sz * 20} m)", fontsize=FS_TITLE, pad=5)
        _sw_contour(ax, labels)

        if c == chosen_idx:
            im_feat_ref = im
            for sp in ax.spines.values():
                sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    # Labels reference panel (last column)
    ax_lab = axes[-1]
    ax_lab.imshow(labels, cmap=cmap_lab, norm=norm_lab, origin="upper")
    ax_lab.set_xticks([]); ax_lab.set_yticks([])
    ax_lab.set_title("Smoothed labels\n(reference)", fontsize=FS_TITLE, pad=5)
    for sp in ax_lab.spines.values():
        sp.set_edgecolor("#555555"); sp.set_linewidth(1.5)

    # Legend
    leg = [Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                  label="sandwave area boundary")]
    axes[chosen_idx].legend(handles=leg, fontsize=FS - 3, loc="lower right",
                            framealpha=0.85, edgecolor="#aaaaaa")

    # Colorbar (below feature columns only)
    fig.canvas.draw()
    cbar_y, cbar_h = 0.055, 0.025
    pos0 = axes[0].get_position()
    pos4 = axes[len(sizes) - 1].get_position()
    cax = fig.add_axes([pos0.x0, cbar_y, pos4.x1 - pos0.x0, cbar_h])
    cb  = fig.colorbar(im_feat_ref, cax=cax, orientation="horizontal")
    cb.set_label("Local std dev  σ(z)  (m)", fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Local standard deviation window size — sandwave feature  "
        f"(chosen: {SWD_STD_SIZE} px = {SWD_STD_SIZE * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 15 — Sandwave detection: gradient features (pre-smoothing sigma comparison)
# ---------------------------------------------------------------------------

def plot_swd_gradient_features(save_path=None):
    """
    Justify the gradient pre-smoothing sigma = 5 px (100 m) used for both
    the local gradient |∇z| and the gradient-of-gradient |∇|∇z|| features.

    Layout: 2 rows × 6 columns  (smooth σ = 1, 2, 5, 10, 20 px + labels reference)
      Row 0  |∇z|          — gradient magnitude after Gaussian smoothing.
      Row 1  |∇|∇z||       — gradient of gradient (second-order variation / curvature).
    Both rows share the same five σ values; the last column shows the sandwave labels.

    Small σ → raw sensor noise amplified by differentiation → speckled gradient maps.
    Large σ → over-smoothed → sandwave-scale slope / curvature signal lost.
    Chosen σ = 5 px (100 m) highlighted with orange border.
    """
    from matplotlib.lines import Line2D

    print("  Loading sandwave raster …", flush=True)
    filled, mean_val, nan_mask = _load_swd()
    labels = _load_swd_labels()
    cmap_lab, norm_lab = _labels_cmap_norm()

    _, residuals = _gaussian_detrend(filled, SWD_DETREND_SIGMA)

    smooths    = [1, 2, 5, 10, 20]
    chosen_idx = smooths.index(SWD_GRAD_SMOOTH)

    all_grad, all_grad2 = [], []
    for sm in smooths:
        g  = _local_grad(residuals, sm);   g[nan_mask]  = np.nan
        g2 = _local_grad(_local_grad(residuals, sm), sm); g2[nan_mask] = np.nan
        all_grad.append(g)
        all_grad2.append(g2)

    # Per-panel 99th-percentile colour limits so smoothed panels are not washed out.
    grad_vmaxes  = [float(np.nanpercentile(g,  99)) for g in all_grad]
    grad2_vmaxes = [float(np.nanpercentile(g2, 99)) for g2 in all_grad2]
    # Reference limits for the colorbars: use the chosen parameter's panel
    grad_max_ref  = grad_vmaxes[chosen_idx]
    grad2_max_ref = grad2_vmaxes[chosen_idx]

    cm_feat = copy.copy(plt.colormaps[CMAP_FEAT]); cm_feat.set_bad(NAN_COLOR)

    n_rows = 2
    n_cols = len(smooths) + 1   # last col = labels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9))
    fig.subplots_adjust(hspace=0.08, wspace=0.06,
                        left=0.09, right=0.97, top=0.88, bottom=0.13)

    im_g_ref = im_g2_ref = None
    for c, (sm, g, g2) in enumerate(zip(smooths, all_grad, all_grad2)):
        ax0, ax1 = axes[0][c], axes[1][c]

        # Row 0 — gradient magnitude (per-panel scale)
        im_g = ax0.imshow(
            np.ma.masked_where(nan_mask, g),
            cmap=cm_feat, origin="upper", vmin=0, vmax=grad_vmaxes[c])
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f"σ = {sm} px  ({sm * 20} m)", fontsize=FS_TITLE, pad=6)
        _sw_contour(ax0, labels)
        if c == 0:
            ax0.set_ylabel("|∇z|  gradient magnitude", fontsize=FS, labelpad=8)

        # Row 1 — gradient of gradient (per-panel scale)
        im_g2 = ax1.imshow(
            np.ma.masked_where(nan_mask, g2),
            cmap=cm_feat, origin="upper", vmin=0, vmax=grad2_vmaxes[c])
        ax1.set_xticks([]); ax1.set_yticks([])
        _sw_contour(ax1, labels)
        if c == 0:
            ax1.set_ylabel("|∇|∇z||  gradient of gradient", fontsize=FS, labelpad=8)

        # Per-panel scale annotation (top-right corner)
        ax0.text(0.97, 0.97, f"max={grad_vmaxes[c]:.3f}",
                 transform=ax0.transAxes, ha="right", va="top",
                 fontsize=FS - 4, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333",
                           alpha=0.75, edgecolor="none"))
        ax1.text(0.97, 0.97, f"max={grad2_vmaxes[c]:.4f}",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=FS - 4, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333",
                           alpha=0.75, edgecolor="none"))

        # Orange border for chosen column
        if c == chosen_idx:
            im_g_ref  = im_g
            im_g2_ref = im_g2
            for r in range(n_rows):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    # Labels reference column (last)
    for r in range(n_rows):
        ax_lab = axes[r][-1]
        ax_lab.imshow(labels, cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax_lab.set_xticks([]); ax_lab.set_yticks([])
        if r == 0:
            ax_lab.set_title("Smoothed labels\n(reference)", fontsize=FS_TITLE, pad=6)
        for sp in ax_lab.spines.values():
            sp.set_edgecolor("#555555"); sp.set_linewidth(1.5)

    # Legend
    leg = [Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                  label="sandwave area boundary")]
    axes[0][chosen_idx].legend(handles=leg, fontsize=FS - 3, loc="lower right",
                                framealpha=0.85, edgecolor="#aaaaaa")

    # Colorbars — one per row, spanning feature columns only
    fig.canvas.draw()
    cbar_y, cbar_h = 0.040, 0.022
    pos0  = axes[0][0].get_position()
    pos_e = axes[0][len(smooths) - 1].get_position()
    feat_w = pos_e.x1 - pos0.x0
    half_w = feat_w * 0.47

    cax0 = fig.add_axes([pos0.x0,              cbar_y, half_w, cbar_h])
    cax1 = fig.add_axes([pos0.x0 + feat_w * 0.53, cbar_y, half_w, cbar_h])

    cb0 = fig.colorbar(im_g_ref,  cax=cax0, orientation="horizontal")
    cb0.set_label(f"|∇z|  (m/px)  — scale shown for chosen σ={SWD_GRAD_SMOOTH} px",
                  fontsize=FS - 1)
    cb0.ax.tick_params(labelsize=FS - 3)

    cb1 = fig.colorbar(im_g2_ref, cax=cax1, orientation="horizontal")
    cb1.set_label(f"|∇|∇z||  (m/px²)  — scale shown for chosen σ={SWD_GRAD_SMOOTH} px",
                  fontsize=FS - 1)
    cb1.ax.tick_params(labelsize=FS - 3)

    # Note about independent scaling
    fig.text(0.97, 0.005, "colour scale is independent per panel  (99th-percentile max)",
             ha="right", va="bottom", fontsize=FS - 4, color="#555555", style="italic")

    fig.suptitle(
        f"Gradient features — pre-smoothing sigma  "
        f"(chosen: σ = {SWD_GRAD_SMOOTH} px = {SWD_GRAD_SMOOTH * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.96)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 20 — Closing: kernel size and threshold comparison (cell 47)
# ---------------------------------------------------------------------------

def plot_swd_closing_effect(save_path=None):
    """
    Show the full pipeline leading into binary closing and the effect of kernel size.

    Layout: 3 rows × 5 columns  (kernel sizes: 1×1, 3×3, 5×5, 7×7, 9×9)
      Row 0  Gaussian-smoothed probability map (σ=20) — shared input, same for all cols.
             The threshold value is marked on the shared colorbar below this row.
      Row 1  Binary after threshold  t=0.35 — the direct input to closing, same for all.
      Row 2  Result after binary closing with that kernel (2 iterations).
             Green contour = chosen full-pipeline boundary.

    Rows 0 and 1 are constant across columns; they show exactly what closing operates on.
    Row 2 shows how kernel size controls how much gap-bridging occurs.
    Orange border = chosen 3×3 column.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    print("  Loading sandwave labels (cell 47) …", flush=True)
    _, _, nan_mask = _load_swd()
    valid_mask  = ~nan_mask
    labels_raw  = _load_swd_raw_labels()
    labels_bin  = np.where(labels_raw < 0, 0, labels_raw)
    labels_ref  = _load_swd_labels()

    kernel_sizes = [1, 3, 5, 7, 9]
    chosen_idx   = kernel_sizes.index(3)

    # ---- Compute the shared smoothed map and binary-threshold input ----
    labels_f = labels_bin.astype(np.float32)
    valid_f  = valid_mask.astype(np.float32)
    weighted = ndimage.gaussian_filter(labels_f * valid_f,
                                       sigma=SWD_CLEAN_SIGMA, mode="nearest")
    support  = ndimage.gaussian_filter(valid_f, sigma=SWD_CLEAN_SIGMA, mode="nearest")
    smooth_f = np.divide(weighted, support,
                         out=np.zeros_like(weighted, dtype=np.float32),
                         where=support > 1e-6)
    smooth_display = np.where(valid_mask, smooth_f.astype(float), np.nan)

    binary = (smooth_f > SWD_CLEAN_THRESH) & valid_mask
    binary_lbl = np.where(~valid_mask, -1, binary.astype(np.int8)).astype(np.int8)

    # ---- Closing results for each kernel size ----
    def _close(ksize):
        struct = np.ones((ksize, ksize), dtype=bool)
        closed = ndimage.binary_closing(binary, structure=struct,
                                        iterations=SWD_CLOSING_ITER, border_value=1)
        out = np.zeros(binary.shape, dtype=np.int8)
        out[closed] = 1; out[~valid_mask] = -1
        return out

    closed_results = [_close(k) for k in kernel_sizes]

    cmap_lab, norm_lab = _labels_cmap_norm()
    cm_smooth = copy.copy(plt.colormaps["YlOrRd"]); cm_smooth.set_bad(NAN_COLOR)

    n_rows, n_cols = 3, len(kernel_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))
    fig.subplots_adjust(hspace=0.08, wspace=0.06,
                        left=0.12, right=0.97, top=0.93, bottom=0.08)

    im_smooth_ref = None
    for c, (k, closed) in enumerate(zip(kernel_sizes, closed_results)):
        ax0, ax1, ax2 = axes[0][c], axes[1][c], axes[2][c]

        # Row 0 — smoothed probability (identical for all cols)
        im_sm = ax0.imshow(np.ma.masked_invalid(smooth_display),
                           cmap=cm_smooth, origin="upper", vmin=0.0, vmax=1.0)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f"kernel  {k}×{k}", fontsize=FS_TITLE, pad=6)

        # Row 1 — binary after threshold (identical for all cols)
        ax1.imshow(binary_lbl, cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax1.set_xticks([]); ax1.set_yticks([])

        # Row 2 — after closing with this kernel
        ax2.imshow(closed, cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax2.set_xticks([]); ax2.set_yticks([])
        _sw_contour(ax2, labels_ref)

        # Orange border on chosen column
        if c == chosen_idx:
            im_smooth_ref = im_sm
            for r in range(n_rows):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    axes[0][0].set_ylabel(f"Smoothed probability\n(σ = {SWD_CLEAN_SIGMA} px)",
                          fontsize=FS - 1, labelpad=8)
    axes[1][0].set_ylabel(f"After threshold  t = {SWD_CLEAN_THRESH}",
                          fontsize=FS - 1, labelpad=8)
    axes[2][0].set_ylabel(f"After closing\n({SWD_CLOSING_ITER} iterations)",
                          fontsize=FS - 1, labelpad=8)

    # Legends
    leg_lab = [Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed"),
               Patch(facecolor="#e53935", label="sandwave"),
               Patch(facecolor=NAN_COLOR, label="no data"),
               Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                      label="chosen full-pipeline result")]
    axes[2][-1].legend(handles=leg_lab, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    # Colorbar for row 0 (probability), with threshold line
    fig.canvas.draw()
    cbar_y, cbar_h = 0.028, 0.020
    pos0  = axes[0][0].get_position()
    pos0e = axes[0][-1].get_position()
    cax   = fig.add_axes([pos0.x0, cbar_y, pos0e.x1 - pos0.x0, cbar_h])
    cb    = fig.colorbar(im_smooth_ref, cax=cax, orientation="horizontal")
    cb.set_label("Smoothed probability", fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 3)
    cb.ax.axvline(x=SWD_CLEAN_THRESH, color="#1565c0", lw=2.0)
    cb.ax.text(SWD_CLEAN_THRESH + 0.015, 0.5,
               f"t = {SWD_CLEAN_THRESH}",
               ha="left", va="center", fontsize=FS - 4, color="#1565c0",
               transform=cb.ax.get_yaxis_transform())

    fig.suptitle(
        f"Binary closing kernel size — pipeline context: smoothed → threshold → close  "
        f"(chosen: 3×3, {SWD_CLOSING_ITER} iterations)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 21 — Erosion component of binary closing (cell 14)
# ---------------------------------------------------------------------------

def plot_swd_erosion_effect(save_path=None):
    """
    Decompose binary_closing = dilation → erosion to make the erosion step visible.
    Uses cell 14 where thin sandwave features are visibly eroded back.

    Layout: 3 rows × 4 columns  (iterations: 1, 2, 5, 10)
      Row 0  After binary *dilation* only  (no erosion yet).
      Row 1  After binary *closing*  (= dilation then erosion).
      Row 2  Erosion effect — pixels removed by erosion highlighted in blue.
             (blue = dilated but NOT in closing result)

    Orange border = chosen iteration count (iter = 2).
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, BoundaryNorm

    print("  Loading sandwave labels (cell 47) …", flush=True)
    _, _, nan_mask = _load_swd()
    valid_mask   = ~nan_mask
    labels_raw   = _load_swd_raw_labels()
    labels_ref   = _load_swd_labels()
    labels_bin   = np.where(labels_raw < 0, 0, labels_raw)

    # Compute smoothed probability + threshold (same params as chosen pipeline)
    labels_f = labels_bin.astype(np.float32)
    valid_f  = valid_mask.astype(np.float32)
    weighted = ndimage.gaussian_filter(labels_f * valid_f,
                                       sigma=SWD_CLEAN_SIGMA, mode="nearest")
    support  = ndimage.gaussian_filter(valid_f, sigma=SWD_CLEAN_SIGMA, mode="nearest")
    smooth_f = np.divide(weighted, support,
                         out=np.zeros_like(weighted, dtype=np.float32),
                         where=support > 1e-6)
    binary_input = (smooth_f > SWD_CLEAN_THRESH) & valid_mask

    struct = np.ones((3, 3), dtype=bool)
    iters  = [1, 2, 5, 10]
    chosen_col = iters.index(SWD_CLOSING_ITER)  # iter=2 → col 1

    # 4-class colormap for Row 2:  -1=gray, 0=white, 1=red(kept), 2=blue(eroded)
    cmap4 = ListedColormap(["#cccccc", "white", "#e53935", "#1565c0"])
    norm4 = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], 4)

    cmap_lab, norm_lab = _labels_cmap_norm()

    n_rows, n_cols = 3, len(iters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 13))
    fig.subplots_adjust(hspace=0.08, wspace=0.06,
                        left=0.14, right=0.97, top=0.93, bottom=0.06)

    for c, n_iter in enumerate(iters):
        dilated = ndimage.binary_dilation(binary_input, structure=struct,
                                          iterations=n_iter)
        closed  = ndimage.binary_erosion(dilated, structure=struct,
                                         iterations=n_iter, border_value=1)
        eroded_away = dilated & ~closed   # added by dilation, removed by erosion

        def _lbl(mask):
            out = np.zeros(binary_input.shape, dtype=np.int8)
            out[mask] = 1; out[~valid_mask] = -1
            return out

        # Row 2 combined: 0=flat, 1=kept sandwave, 2=eroded away, -1=nodata
        diff_img = np.where(~valid_mask, -1,
                   np.where(closed,      1,
                   np.where(eroded_away, 2, 0))).astype(np.int8)

        ax0, ax1, ax2 = axes[0][c], axes[1][c], axes[2][c]

        ax0.imshow(_lbl(dilated), cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax1.imshow(_lbl(closed),  cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax2.imshow(diff_img,      cmap=cmap4,    norm=norm4,    origin="upper")

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])
            _sw_contour(ax, labels_ref)

        axes[0][c].set_title(f"{n_iter} iter{'s' if n_iter > 1 else ''}",
                             fontsize=FS_TITLE, pad=6)

        if c == chosen_col:
            for r in range(n_rows):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    axes[0][0].set_ylabel("After dilation only", fontsize=FS, labelpad=8)
    axes[1][0].set_ylabel("After closing\n(dilation + erosion)", fontsize=FS, labelpad=8)
    axes[2][0].set_ylabel("Erosion effect\n(blue = removed by erosion)", fontsize=FS, labelpad=8)

    # Legends
    leg_lab = [Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed"),
               Patch(facecolor="#e53935", label="sandwave (kept)"),
               Patch(facecolor=NAN_COLOR, label="no data"),
               Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                      label="chosen full-pipeline boundary")]
    axes[0][-1].legend(handles=leg_lab, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    leg_ero = [Patch(facecolor="#e53935", label="sandwave (survived erosion)"),
               Patch(facecolor="#1565c0", label="removed by erosion"),
               Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed"),
               Patch(facecolor=NAN_COLOR, label="no data")]
    axes[2][-1].legend(handles=leg_ero, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    fig.suptitle(
        f"Erosion component of binary closing — "
        f"cell {SWD_CELL}, CDI {SWD_CDI}  "
        f"(3×3 kernel, border_value=1)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 16 — Sandwave detection: data overview
# ---------------------------------------------------------------------------

def plot_swd_data_overview(save_path=None):
    """
    Two-panel overview of the sandwave detection example raster.

    Left   Destriped bathymetry (cmocean.deep) with NaN areas in gray and a 1 km scale bar.
    Right  Same bathymetry with sandwave classification overlaid:
             semi-transparent red fill = detected sandwave area;
             green contour = boundary of smoothed labels.
    """
    from matplotlib.patches import Patch

    print("  Loading sandwave raster …", flush=True)
    filled, mean_val, nan_mask = _load_swd()
    labels = _load_swd_labels()

    raster = filled + mean_val   # absolute depth (m)
    h, w   = raster.shape

    vmin = float(np.nanmin(raster[~nan_mask]))
    vmax = float(np.nanmax(raster[~nan_mask]))

    cm_bath = copy.copy(CMAP_BATH); cm_bath.set_bad(NAN_COLOR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(wspace=0.06, left=0.04, right=0.94, top=0.89, bottom=0.12)

    bath_masked = np.ma.masked_where(nan_mask, raster)

    # ---- Left panel: bathymetry only ----------------------------------------
    im = axes[0].imshow(bath_masked, cmap=cm_bath, origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title("Destriped bathymetry", fontsize=FS_TITLE, pad=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # 1 km scale bar (bottom-left corner)
    scale_px = 1000 / 20   # 1 km at 20 m/px
    sx0 = w * 0.05
    sy  = h * 0.93
    axes[0].plot([sx0, sx0 + scale_px], [sy, sy],
                 "-", color="white", lw=3, solid_capstyle="butt")
    axes[0].text(sx0 + scale_px / 2, sy + h * 0.025,
                 "1 km", color="white", ha="center", va="top",
                 fontsize=FS - 2, fontweight="bold")

    # ---- Right panel: bathymetry + classification overlay -------------------
    axes[1].imshow(bath_masked, cmap=cm_bath, origin="upper", vmin=vmin, vmax=vmax)

    # Semi-transparent red fill over sandwave pixels
    sw_fill = np.where(labels == 1, 1.0, np.nan)
    axes[1].imshow(np.ma.masked_invalid(sw_fill),
                   cmap="Reds", origin="upper", alpha=0.45, vmin=0, vmax=1)
    _sw_contour(axes[1], labels)

    axes[1].set_title("Sandwave classification overlay", fontsize=FS_TITLE, pad=7)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Legend for right panel
    leg = [Patch(facecolor="#e53935", alpha=0.55, label="sandwave area"),
           Patch(facecolor=NAN_COLOR, label="no data")]
    axes[1].legend(handles=leg, fontsize=FS - 2, loc="lower right",
                   framealpha=0.85, edgecolor="#aaaaaa")

    # Shared colorbar (spans both panels)
    fig.canvas.draw()
    pos0  = axes[0].get_position()
    pos1  = axes[1].get_position()
    cax   = fig.add_axes([pos0.x0, 0.045, pos1.x1 - pos0.x0, 0.025])
    cb    = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Depth (m)", fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"Example raster — cell {SWD_CELL}, CDI {SWD_CDI}  "
        f"({w * 20 / 1000:.0f} × {h * 20 / 1000:.0f} km,  20 m/px)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 17 — Label cleaning pipeline: step-by-step stages
# ---------------------------------------------------------------------------

def plot_swd_label_pipeline(save_path=None):
    """
    Six-panel step-by-step illustration of clean_smoothed_labels at chosen params.
    Uses cell 24 / CDI 2174760, which has visible changes at every cleaning step
    (43 px added by closing, 898 px added by fill_holes, 148 px removed as small islands).

    Panels (left → right):
      1 Raw K-Means labels
      2 Gaussian-smoothed probability map  (σ = 20 px)
      3 After threshold  t = 0.35
      4 After binary closing  (2 iter, 3×3)   — green overlay = newly added pixels
      5 After fill_holes                       — green overlay = newly filled pixels
      6 After removing small islands (<200 px) — blue  overlay = removed pixels

    The threshold value is marked as a vertical line on the smoothed-map colorbar.
    """
    from matplotlib.patches import Patch

    PIPE_CELL = 24
    PIPE_CDI  = 2174760
    raster_path     = f"destriped_rasters/cell_{PIPE_CELL}_CDI_{PIPE_CDI}_destriped.npy"
    raw_labels_path = (f"sandwave_detection_v8/labels/"
                       f"cell_{PIPE_CELL}_CDI_{PIPE_CDI}_destriped_labels.npy")

    print("  Loading sandwave labels …", flush=True)
    raster     = np.load(raster_path)
    nan_mask   = np.isnan(raster)
    valid_mask = ~nan_mask
    labels_raw = np.load(raw_labels_path)
    labels_bin = np.where(labels_raw < 0, 0, labels_raw)

    steps = _clean_steps(labels_bin, valid_mask,
                         sigma=SWD_CLEAN_SIGMA, threshold=SWD_CLEAN_THRESH,
                         closing_iter=SWD_CLOSING_ITER, min_pixels=SWD_MIN_PIXELS)

    thr   = steps["after_threshold"]
    clos  = steps["after_closing"]
    fill  = steps["after_fillholes"]
    final = steps["final"]

    add_closing   = (clos == 1) & (thr  != 1)   # pixels added by closing
    add_fillholes = (fill == 1) & (clos != 1)   # pixels added by fill_holes
    rem_final     = (fill == 1) & (final != 1)  # pixels removed by small-island filter

    cmap_lab, norm_lab = _labels_cmap_norm()
    cm_smooth = copy.copy(plt.colormaps["YlOrRd"]); cm_smooth.set_bad(NAN_COLOR)

    panels = [
        ("Raw K-Means labels",
         labels_raw, cmap_lab, norm_lab, None, None),
        (f"Gaussian smoothing\n(σ = {SWD_CLEAN_SIGMA} px)",
         steps["smoothed_float"], cm_smooth, None, 0.0, 1.0),
        (f"Threshold  t = {SWD_CLEAN_THRESH}",
         thr, cmap_lab, norm_lab, None, None),
        (f"Binary closing\n({SWD_CLOSING_ITER} iter, 3×3)",
         clos, cmap_lab, norm_lab, None, None),
        ("Fill holes",
         fill, cmap_lab, norm_lab, None, None),
        (f"Remove small islands\n(< {SWD_MIN_PIXELS} px)",
         final, cmap_lab, norm_lab, None, None),
    ]

    # RGBA overlays: green = added, blue = removed
    _green = [0.00, 0.80, 0.20, 0.88]
    _blue  = [0.08, 0.39, 0.75, 0.88]
    change_overlays = [None, None, None, (add_closing, _green), (add_fillholes, _green), (rem_final, _blue)]

    fig, axes = plt.subplots(1, 6, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.97, top=0.84, bottom=0.17)

    im_smooth = None
    for c, ((title, data, cmap, norm, vmin, vmax), overlay) in enumerate(
            zip(panels, change_overlays)):
        ax = axes[c]
        if vmin is not None:
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, origin="upper",
                           vmin=vmin, vmax=vmax)
            im_smooth = im
        else:
            im = ax.imshow(data, cmap=cmap, norm=norm, origin="upper")

        if overlay is not None:
            mask, rgba = overlay
            ov = np.zeros((*mask.shape, 4), dtype=float)
            ov[mask] = rgba
            ax.imshow(ov, origin="upper")

        ax.set_title(title, fontsize=FS_TITLE, pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    # Arrows between panels
    for c in range(5):
        axes[c].annotate("", xy=(1.04, 0.5), xycoords="axes fraction",
                         xytext=(1.0,  0.5), textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # Colorbar below the smoothed panel, with threshold marker
    fig.canvas.draw()
    pos1 = axes[1].get_position()
    cax  = fig.add_axes([pos1.x0, 0.055, pos1.width, 0.025])
    cb   = fig.colorbar(im_smooth, cax=cax, orientation="horizontal")
    cb.set_label("Smoothed probability", fontsize=FS - 2)
    cb.ax.tick_params(labelsize=FS - 3)
    cb.ax.axvline(x=SWD_CLEAN_THRESH, color="#1565c0", lw=2.0)
    cb.ax.text(SWD_CLEAN_THRESH + 0.03, 0.5,
               f"t = {SWD_CLEAN_THRESH}",
               ha="left", va="center", fontsize=FS - 4, color="#1565c0",
               transform=cb.ax.get_yaxis_transform())

    # Legend
    leg = [Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed (0)"),
           Patch(facecolor="#e53935", label="sandwave (1)"),
           Patch(facecolor=NAN_COLOR, label="no data (−1)"),
           Patch(facecolor="#00cc33", label="added by step"),
           Patch(facecolor="#1465c0", label="removed by step")]
    axes[-1].legend(handles=leg, fontsize=FS - 3, loc="lower right",
                    framealpha=0.85, edgecolor="#aaaaaa")

    fig.suptitle(
        f"Label cleaning pipeline: raw K-Means → smoothed binary classification"
        f"  (cell {PIPE_CELL}, CDI {PIPE_CDI})",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 17 — Label cleaning: Gaussian smoothing sigma comparison
# ---------------------------------------------------------------------------

def plot_swd_sigma_comparison(save_path=None):
    """
    Justify the Gaussian smoothing sigma = 20 px for label cleanup.

    Layout: 2 rows × 5 columns  (σ = 3, 7, 10, 20, 40 px)
      Row 0  Continuous probability map (0–1) after Gaussian smoothing.
             Threshold line shown on the shared colorbar.
      Row 1  Final binary labels after the full cleaning pipeline.
             Green contour = boundary produced by the chosen σ = 20.

    Small σ → probability map noisy, fragmented sandwave patches.
    Large σ → probability map over-smooth, sandwave regions bloat and merge.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    print("  Loading sandwave labels …", flush=True)
    _, _, nan_mask = _load_swd()
    valid_mask  = ~nan_mask
    labels_raw  = _load_swd_raw_labels()
    labels_bin  = np.where(labels_raw < 0, 0, labels_raw)
    labels_ref  = _load_swd_labels()   # chosen result for contour overlay

    sigmas     = [3, 7, 10, 20, 40]
    chosen_idx = sigmas.index(SWD_CLEAN_SIGMA)

    all_smooth, all_final = [], []
    for s in sigmas:
        st = _clean_steps(labels_bin, valid_mask,
                          sigma=s, threshold=SWD_CLEAN_THRESH,
                          closing_iter=SWD_CLOSING_ITER, min_pixels=SWD_MIN_PIXELS)
        all_smooth.append(st["smoothed_float"])
        all_final.append(st["final"])

    cmap_lab, norm_lab = _labels_cmap_norm()
    cm_smooth = copy.copy(plt.colormaps["YlOrRd"]); cm_smooth.set_bad(NAN_COLOR)

    n_rows, n_cols = 2, len(sigmas)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9))
    fig.subplots_adjust(hspace=0.08, wspace=0.06,
                        left=0.09, right=0.97, top=0.88, bottom=0.13)

    im_smooth_ref = None
    for c, (s, sm, fin) in enumerate(zip(sigmas, all_smooth, all_final)):
        ax0, ax1 = axes[0][c], axes[1][c]

        # Row 0 — continuous probability map
        im_sm = ax0.imshow(np.ma.masked_invalid(sm), cmap=cm_smooth,
                           origin="upper", vmin=0.0, vmax=1.0)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f"σ = {s} px  ({s * 20} m)", fontsize=FS_TITLE, pad=6)
        if c == 0:
            ax0.set_ylabel("Smoothed probability", fontsize=FS, labelpad=8)

        # Row 1 — final binary labels
        ax1.imshow(fin, cmap=cmap_lab, norm=norm_lab, origin="upper")
        ax1.set_xticks([]); ax1.set_yticks([])
        _sw_contour(ax1, labels_ref)
        if c == 0:
            ax1.set_ylabel("Final binary labels", fontsize=FS, labelpad=8)

        # Orange border for chosen column
        if c == chosen_idx:
            im_smooth_ref = im_sm
            for r in range(n_rows):
                for sp in axes[r][c].spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

    # Legends
    from matplotlib.patches import Patch
    leg_lab = [Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed"),
               Patch(facecolor="#e53935", label="sandwave"),
               Patch(facecolor=NAN_COLOR, label="no data")]
    axes[1][0].legend(handles=leg_lab, fontsize=FS - 3, loc="lower left",
                      framealpha=0.85, edgecolor="#aaaaaa")

    leg_cnt = [plt.matplotlib.lines.Line2D(
        [0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
        label=f"chosen boundary  (σ = {SWD_CLEAN_SIGMA} px)")]
    axes[1][-1].legend(handles=leg_cnt, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    # Colorbar below row 0 (probability); mark threshold
    fig.canvas.draw()
    cbar_y, cbar_h = 0.042, 0.022
    pos0  = axes[0][0].get_position()
    pos0e = axes[0][-1].get_position()
    full_w = pos0e.x1 - pos0.x0

    cax = fig.add_axes([pos0.x0, cbar_y, full_w, cbar_h])
    cb  = fig.colorbar(im_smooth_ref, cax=cax, orientation="horizontal")
    cb.set_label("Smoothed probability", fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 3)
    cb.ax.axvline(x=SWD_CLEAN_THRESH, color="#1565c0", lw=2.0)
    cb.ax.text(SWD_CLEAN_THRESH + 0.015, 0.5,
               f"t = {SWD_CLEAN_THRESH}",
               ha="left", va="center", fontsize=FS - 4, color="#1565c0",
               transform=cb.ax.get_yaxis_transform())

    fig.suptitle(
        f"Gaussian smoothing sigma — label cleaning  "
        f"(chosen: σ = {SWD_CLEAN_SIGMA} px = {SWD_CLEAN_SIGMA * 20} m)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.96)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 18 — Label cleaning: threshold, closing, and min-size comparisons
# ---------------------------------------------------------------------------

def plot_swd_threshold_morphology(save_path=None):
    """
    Justify threshold = 0.35, closing_iter = 2, and min_cluster_pixels = 200.

    Layout: 3 rows × 5 columns — each row varies one parameter, others fixed.
      Row 0  Threshold     [0.05, 0.15, 0.35, 0.50, 0.70]
      Row 1  Closing iter  [0, 1, 2, 5, 10]
      Row 2  Min size (px) [0, 50, 100, 200, 500]

    Each cell shows the *final* binary labels for that parameter combination.
    Green contour = reference boundary (all params at chosen values).
    Orange border = chosen value per row.

    Row 0 story: low t → too much area; high t → sandwaves missed.
    Row 1 story: 0 iter → fragmented patches with gaps; high iter → regions merge.
    Row 2 story: 0 px → many noise islands; large min → real patches removed.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    print("  Loading sandwave labels …", flush=True)
    _, _, nan_mask = _load_swd()
    valid_mask  = ~nan_mask
    labels_raw  = _load_swd_raw_labels()
    labels_bin  = np.where(labels_raw < 0, 0, labels_raw)
    labels_ref  = _load_swd_labels()

    thresh_vals = [0.05, 0.15, 0.35, 0.50, 0.70]
    close_vals  = [0, 1, 2, 5, 10]
    minpx_vals  = [0, 50, 100, 200, 500]

    chosen_cols = [
        thresh_vals.index(SWD_CLEAN_THRESH),   # row 0
        close_vals.index(SWD_CLOSING_ITER),    # row 1
        minpx_vals.index(SWD_MIN_PIXELS),      # row 2
    ]

    # Pre-compute final labels for each row
    rows_data = [
        [_clean_steps(labels_bin, valid_mask,
                      sigma=SWD_CLEAN_SIGMA, threshold=t,
                      closing_iter=SWD_CLOSING_ITER, min_pixels=SWD_MIN_PIXELS)["final"]
         for t in thresh_vals],
        [_clean_steps(labels_bin, valid_mask,
                      sigma=SWD_CLEAN_SIGMA, threshold=SWD_CLEAN_THRESH,
                      closing_iter=ci, min_pixels=SWD_MIN_PIXELS)["final"]
         for ci in close_vals],
        [_clean_steps(labels_bin, valid_mask,
                      sigma=SWD_CLEAN_SIGMA, threshold=SWD_CLEAN_THRESH,
                      closing_iter=SWD_CLOSING_ITER, min_pixels=mp)["final"]
         for mp in minpx_vals],
    ]
    rows_vals = [thresh_vals, close_vals, minpx_vals]
    rows_labels = [
        [f"t = {v}" for v in thresh_vals],
        [f"{v} iter" for v in close_vals],
        [f"{v} px"   for v in minpx_vals],
    ]
    rows_ylabel = [
        f"Threshold  t\n(σ={SWD_CLEAN_SIGMA}, close={SWD_CLOSING_ITER}, min={SWD_MIN_PIXELS} px)",
        f"Closing iterations\n(σ={SWD_CLEAN_SIGMA}, t={SWD_CLEAN_THRESH}, min={SWD_MIN_PIXELS} px)",
        f"Min island size (px)\n(σ={SWD_CLEAN_SIGMA}, t={SWD_CLEAN_THRESH}, close={SWD_CLOSING_ITER})",
    ]

    cmap_lab, norm_lab = _labels_cmap_norm()

    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))
    fig.subplots_adjust(hspace=0.10, wspace=0.06,
                        left=0.14, right=0.97, top=0.94, bottom=0.06)

    for r, (data_row, val_row, lbl_row, chosen_c, ylabel) in enumerate(
            zip(rows_data, rows_vals, rows_labels, chosen_cols, rows_ylabel)):
        for c, (data, lbl) in enumerate(zip(data_row, lbl_row)):
            ax = axes[r][c]
            ax.imshow(data, cmap=cmap_lab, norm=norm_lab, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            _sw_contour(ax, labels_ref)

            # Per-panel parameter value (top centre)
            ax.text(0.5, 0.97, lbl,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=FS - 2,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.85, edgecolor="none"))

            # Orange border for chosen column in this row
            if c == chosen_c:
                for sp in ax.spines.values():
                    sp.set_edgecolor(HIGHLIGHT_COLOR); sp.set_linewidth(3)

        axes[r][0].set_ylabel(ylabel, fontsize=FS - 1, labelpad=8)

    # Shared legend (top-right of last panel in first row)
    leg = [Patch(facecolor="white",   edgecolor="#aaaaaa", label="flat seabed"),
           Patch(facecolor="#e53935", label="sandwave"),
           Patch(facecolor=NAN_COLOR, label="no data"),
           Line2D([0], [0], color=SW_CONTOUR_COLOR, lw=1.5,
                  label="chosen result boundary")]
    axes[0][-1].legend(handles=leg, fontsize=FS - 3, loc="lower right",
                       framealpha=0.85, edgecolor="#aaaaaa")

    fig.suptitle(
        f"Label post-processing: threshold / closing / min island size  "
        f"(chosen: t={SWD_CLEAN_THRESH}, close={SWD_CLOSING_ITER}, min={SWD_MIN_PIXELS} px)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=0.97)

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 22 — Study-area coverage + sandwave detections overview
# ---------------------------------------------------------------------------

def plot_coverage_and_sandwaves(save_path=None):
    """
    Single-panel overview of the full study area.

    Base layer  : combined destriped bathymetry (cmocean.deep), NaN cells in gray.
    Overlay     : sandwave probability as a semi-transparent red tint —
                  prob = 0 → fully transparent; prob = 1 → solid red (alpha 0.75).
                  The probability per 100 m cell is the mean fraction of surveys
                  that detected a sandwave there (after 5× max-pool downscale).
    Routes      : Route 1 and Route 2 from shapefiles, drawn as solid lines.
    """
    import skimage as ski
    import costmap as cm_mod
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    LABELS_DIR  = "sandwave_detection_v8/labels"
    MAX_ALPHA   = 0.75   # alpha when sandwave probability = 1.0

    def _stack_mean(rasters):
        stacked = np.stack(rasters, axis=0)
        with np.errstate(invalid="ignore"):
            return np.nanmean(stacked, axis=0)

    # ---- Build bathymetry base map ------------------------------------------
    print("  Building bathymetry base map …", flush=True)
    bath = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    raster_dict: dict = {}
    for fname in os.listdir("destriped_rasters"):
        cid = fname.split("_")[1]
        arr = np.load(os.path.join("destriped_rasters", fname))[::-1, :]
        raster_dict.setdefault(cid, []).append(arr)
    for cid, rasters in raster_dict.items():
        avg      = _stack_mean(rasters)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        xs, xe, ys, ye = bath.slice_cost_map(int(cid))
        bath.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    # ---- Build sandwave probability map -------------------------------------
    print("  Building sandwave probability map …", flush=True)
    sw = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    sw_dict: dict = {}
    for fname in [f for f in os.listdir(LABELS_DIR)
                  if f.endswith("destriped_labels_smoothed.npy")]:
        cid = fname.split("_")[1]
        arr = np.load(os.path.join(LABELS_DIR, fname))[::-1, :].astype(float)
        arr[arr == -1] = np.nan
        sw_dict.setdefault(cid, []).append(arr)
    for cid, rasters in sw_dict.items():
        avg      = _stack_mean(rasters)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.max)
        xs, xe, ys, ye = sw.slice_cost_map(int(cid))
        sw.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    # ---- RGBA overlay: red with per-pixel alpha = probability ---------------
    sw_prob = np.where(np.isnan(sw.costs), 0.0, sw.costs)  # NaN → transparent
    sw_prob = np.clip(sw_prob, 0.0, 1.0)
    rgba_overlay        = np.zeros((*sw_prob.shape, 4), dtype=float)
    rgba_overlay[..., 0] = 0.84   # R  (matches #d62728)
    rgba_overlay[..., 1] = 0.15   # G
    rgba_overlay[..., 2] = 0.16   # B
    rgba_overlay[..., 3] = sw_prob * MAX_ALPHA

    # ---- Load routes via geopandas for smooth vector lines ------------------
    try:
        import geopandas as gpd
        USE_VECTOR_ROUTES = True
    except ImportError:
        USE_VECTOR_ROUTES = False

    # ---- Extent in UTM 31N --------------------------------------------------
    bl      = bath.bl          # (555652, 5910512)
    nx, ny  = 80_000, 35_000   # metres
    extent  = [bl[0], bl[0] + nx, bl[1], bl[1] + ny]

    # ---- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.13)

    # Base: bathymetry
    bath_masked = np.ma.masked_invalid(bath.costs)
    cm_bath2 = copy.copy(CMAP_BATH)
    cm_bath2.set_bad(NAN_COLOR)
    im = ax.imshow(
        bath_masked,
        cmap=cm_bath2,
        origin="lower", extent=extent,
        interpolation="nearest",
        aspect="equal",
        zorder=1,
    )

    # Colorbar for bathymetry
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("Depth  [m]", fontsize=FS - 2)
    cbar.ax.tick_params(labelsize=FS - 3)

    # Sandwave overlay (RGBA, per-pixel alpha)
    ax.imshow(
        rgba_overlay,
        origin="lower", extent=extent,
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # ---- Route lines --------------------------------------------------------
    ROUTE_FILES  = ["shapes/line1.shp", "shapes/line2.shp"]
    ROUTE_COLORS = ["#1f77b4", "#ff7f0e"]
    ROUTE_LABELS = ["Route 1", "Route 2"]

    if USE_VECTOR_ROUTES:
        for fpath, color, label in zip(ROUTE_FILES, ROUTE_COLORS, ROUTE_LABELS):
            if os.path.exists(fpath):
                gdf = gpd.read_file(fpath).to_crs(bath.csr)
                gdf.plot(ax=ax, color=color, linewidth=2.0,
                         label=label, zorder=5)
    else:
        for route, color, label in zip(bath.routes, ROUTE_COLORS, ROUTE_LABELS):
            ys_r, xs_r = np.where(route)
            utm_x = bl[0] + xs_r * 100 + 50
            utm_y = bl[1] + ys_r * 100 + 50
            ax.scatter(utm_x, utm_y, c=color, s=2, zorder=5,
                       label=label, linewidths=0)

    # ---- Axes ---------------------------------------------------------------
    ax.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS)
    ax.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=6)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 2)
    plt.setp(ax.get_yticklabels(), fontsize=FS - 2)
    ax.set_title(
        "Destriped bathymetry with sandwave detections",
        fontsize=FS_TITLE + 1, fontweight="bold", pad=10,
    )

    # ---- Legend -------------------------------------------------------------
    # Sandwave probability is shown via a gradient patch (3 alpha steps)
    sw_legend_patches = [
        Patch(facecolor=(0.84, 0.15, 0.16, a), edgecolor="none",
              label=lbl)
        for a, lbl in [
            (MAX_ALPHA * 0.33, "Sandwave prob. < 33 %"),
            (MAX_ALPHA * 0.67, "Sandwave prob. 33–67 %"),
            (MAX_ALPHA * 1.00, "Sandwave prob. > 67 %"),
        ]
    ]
    legend_handles = [
        Patch(facecolor=NAN_COLOR, edgecolor="none", label="No survey data"),
        *sw_legend_patches,
        Line2D([0], [0], color=ROUTE_COLORS[0], lw=2, label=ROUTE_LABELS[0]),
        Line2D([0], [0], color=ROUTE_COLORS[1], lw=2, label=ROUTE_LABELS[1]),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=FS - 2, frameon=True, edgecolor="#cccccc")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 23 — Variance cost map + threshold justification
# ---------------------------------------------------------------------------

def plot_variance_costmap(save_path=None):
    """
    Two-row figure supporting the variance threshold choice (var_threshold = 0.05).

    Row 0  Left  : Continuous log-variance map across the study area.
    Row 0  Right : Histogram of log-variance values with all candidate threshold
                   lines; chosen threshold (0.05) highlighted in orange.
    Row 1         : Binary "high-variance" masks at five candidate thresholds;
                    chosen panel has an orange border and fraction-flagged badge.
    """
    import skimage as ski
    import costmap as cm_mod
    from matplotlib.colors import ListedColormap, BoundaryNorm

    VAR_DIR    = "variance_rasters/Rasters"
    CHOSEN     = 0.05
    THRESHOLDS = [0.02, 0.035, 0.05, 0.08, 0.15]

    # ---- Build log-variance map ---------------------------------------------
    print("  Building variance map …", flush=True)
    cmap_obj = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    d: dict = {}
    for fname in os.listdir(VAR_DIR):
        cid = fname.split("_")[1]
        arr = np.load(os.path.join(VAR_DIR, fname))[::-1, :]
        d.setdefault(cid, []).append(arr)
    for cid, rasters in d.items():
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(np.stack(rasters, axis=0), axis=0)
        rescaled    = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        transformed = np.log10(np.sqrt(np.maximum(rescaled, 0.0)) + 1.0)
        xs, xe, ys, ye = cmap_obj.slice_cost_map(int(cid))
        cmap_obj.add_cost(xs, ys, cost=transformed, x_idx_end=xe, y_idx_end=ye)

    values   = cmap_obj.costs
    has_data = ~np.isnan(values)
    valid    = values[has_data]
    bl       = cmap_obj.bl
    extent   = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]
    vmax_map = float(np.nanpercentile(values, 99))

    # ---- Layout (constrained_layout handles equal-aspect axes cleanly) ------
    # 5 columns: map=4, histogram=1; bottom row 5 panels each 1 col → full width
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs  = fig.add_gridspec(2, 5,
                           height_ratios=[2.2, 1.6],
                           hspace=0.12, wspace=0.15)
    ax_map  = fig.add_subplot(gs[0, :4])
    ax_hist = fig.add_subplot(gs[0,  4])
    ax_t    = [fig.add_subplot(gs[1, i]) for i in range(5)]
    ax_map.set_anchor("W")   # keep map left-aligned when aspect="equal" shrinks it

    # ---- Continuous map -----------------------------------------------------
    cm_var = copy.copy(plt.colormaps["YlOrRd"])
    cm_var.set_bad(NAN_COLOR)
    im = ax_map.imshow(
        np.ma.masked_invalid(values),
        cmap=cm_var, vmin=0, vmax=vmax_map,
        origin="lower", extent=extent,
        interpolation="nearest", aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax_map, shrink=0.9, pad=0.02)
    cbar.set_label(r"$\log_{10}(\sqrt{\mathrm{var}}+1)$  [–]", fontsize=FS - 2)
    cbar.ax.tick_params(labelsize=FS - 3)
    cbar.ax.axhline(y=CHOSEN / vmax_map, color=HIGHLIGHT_COLOR, lw=2.0)
    ax_map.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS - 2)
    ax_map.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS - 2)
    ax_map.locator_params(axis="x", nbins=5)
    ax_map.locator_params(axis="y", nbins=5)
    plt.setp(ax_map.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 3)
    plt.setp(ax_map.get_yticklabels(), fontsize=FS - 3)
    ax_map.set_title(r"Continuous log-variance  $\log_{10}(\sqrt{\sigma^2}+1)$",
                     fontsize=FS_TITLE, pad=6)

    # ---- Histogram ----------------------------------------------------------
    med_v = float(np.median(valid))
    std_v = float(np.std(valid))

    ax_hist.hist(valid, bins=60, color="#888888", edgecolor="none",
                 density=True, orientation="vertical")
    ax_hist.axvline(med_v, color="#333333", lw=1.2, ls=":", zorder=2,
                    label=f"median")
    for thresh in THRESHOLDS:
        is_chosen = (thresh == CHOSEN)
        nsig = (thresh - med_v) / std_v
        ax_hist.axvline(thresh,
                        color=HIGHLIGHT_COLOR if is_chosen else "#444444",
                        lw=2.2 if is_chosen else 1.2,
                        ls="-"  if is_chosen else "--",
                        zorder=3 if is_chosen else 2,
                        label=f"{thresh:.3f}  ({nsig:+.1f}σ)"
                              + (" ✓" if is_chosen else ""))
    # Secondary x-axis on top showing σ from median
    ax_top = ax_hist.secondary_xaxis(
        "top",
        functions=(lambda x: (x - med_v) / std_v,
                   lambda z: z * std_v + med_v),
    )
    ax_top.set_xlabel("σ from median", fontsize=FS - 4)
    ax_top.tick_params(labelsize=FS - 4)
    ax_hist.set_xlabel(r"$\log_{10}(\sqrt{\sigma^2}+1)$", fontsize=FS - 3)
    ax_hist.set_ylabel("Density", fontsize=FS - 3)
    ax_hist.set_title("Value distribution", fontsize=FS_TITLE - 1, pad=6)
    ax_hist.legend(fontsize=FS - 4, title="Threshold  (σ from median)",
                   title_fontsize=FS - 5, frameon=True)
    ax_hist.tick_params(labelsize=FS - 3)

    # ---- Threshold comparison panels ----------------------------------------
    cmap_bin = ListedColormap([NAN_COLOR, "white", "#d62728"])
    norm_bin = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    for ax, thresh in zip(ax_t, THRESHOLDS):
        binary                  = np.full_like(values, np.nan)
        binary[has_data]        = np.where(values[has_data] > thresh, 1.0, 0.0)
        ax.imshow(
            np.ma.masked_invalid(binary),
            cmap=cmap_bin, norm=norm_bin,
            origin="lower", extent=extent,
            interpolation="nearest",
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {thresh:.3f}", fontsize=FS_TITLE - 1, pad=4)
        if thresh == CHOSEN:
            for spine in ax.spines.values():
                spine.set_edgecolor(HIGHLIGHT_COLOR)
                spine.set_linewidth(3.5)
        frac = float(np.mean(binary[has_data] == 1.0)) * 100
        ax.text(0.97, 0.03, f"{frac:.1f} % flagged",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS - 4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.85))

    fig.suptitle(
        f"Variance-based dynamic-seabed cost map  —  chosen threshold = {CHOSEN}",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 24 — Amplitude cost map + threshold justification
# ---------------------------------------------------------------------------

def plot_amplitude_costmap(save_path=None):
    """
    Two-row figure supporting the amplitude threshold choice (amp_threshold = 0.08).

    Row 0  Left  : Continuous sandwave-amplitude map across the study area.
    Row 0  Right : Histogram of amplitude values with all candidate threshold
                   lines; chosen threshold (0.08 m) highlighted in orange.
    Row 1         : Binary "high-amplitude" masks at five candidate thresholds;
                    chosen panel has an orange border and fraction-flagged badge.
    """
    import skimage as ski
    import costmap as cm_mod
    from matplotlib.colors import ListedColormap, BoundaryNorm

    AMP_DIR    = "amplitude_rasters/Rasters_amp"
    CHOSEN     = 0.08
    THRESHOLDS = [0.03, 0.05, 0.08, 0.12, 0.20]

    # ---- Build amplitude map ------------------------------------------------
    print("  Building amplitude map …", flush=True)
    cmap_obj = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    d: dict = {}
    for fname in os.listdir(AMP_DIR):
        cid = fname.split("_")[1]
        arr = np.load(os.path.join(AMP_DIR, fname))[::-1, :]
        d.setdefault(cid, []).append(arr)
    for cid, rasters in d.items():
        with np.errstate(invalid="ignore"):
            avg = np.nanmax(np.stack(rasters, axis=0), axis=0)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.nanmax)
        xs, xe, ys, ye = cmap_obj.slice_cost_map(int(cid))
        cmap_obj.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    values   = cmap_obj.costs
    has_data = ~np.isnan(values)
    valid    = values[has_data]
    bl       = cmap_obj.bl
    extent   = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]
    vmax_map = float(np.nanpercentile(values, 99))

    # ---- Layout (constrained_layout handles equal-aspect axes cleanly) ------
    # 5 columns: map=4, histogram=1; bottom row 5 panels each 1 col → full width
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs  = fig.add_gridspec(2, 5,
                           height_ratios=[2.2, 1.6],
                           hspace=0.12, wspace=0.15)
    ax_map  = fig.add_subplot(gs[0, :4])
    ax_hist = fig.add_subplot(gs[0,  4])
    ax_t    = [fig.add_subplot(gs[1, i]) for i in range(5)]
    ax_map.set_anchor("W")   # keep map left-aligned when aspect="equal" shrinks it

    # ---- Continuous map -----------------------------------------------------
    cm_amp = copy.copy(plt.colormaps["YlOrBr"])
    cm_amp.set_bad(NAN_COLOR)
    im = ax_map.imshow(
        np.ma.masked_invalid(values),
        cmap=cm_amp, vmin=0, vmax=vmax_map,
        origin="lower", extent=extent,
        interpolation="nearest", aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax_map, shrink=0.9, pad=0.02)
    cbar.set_label("Sandwave amplitude  [m]", fontsize=FS - 2)
    cbar.ax.tick_params(labelsize=FS - 3)
    cbar.ax.axhline(y=CHOSEN / vmax_map, color=HIGHLIGHT_COLOR, lw=2.0)
    ax_map.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS - 2)
    ax_map.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS - 2)
    ax_map.locator_params(axis="x", nbins=5)
    ax_map.locator_params(axis="y", nbins=5)
    plt.setp(ax_map.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 3)
    plt.setp(ax_map.get_yticklabels(), fontsize=FS - 3)
    ax_map.set_title("Continuous sandwave-amplitude map  (max across surveys)",
                     fontsize=FS_TITLE, pad=6)

    # ---- Histogram ----------------------------------------------------------
    med_v = float(np.median(valid))
    std_v = float(np.std(valid))

    ax_hist.hist(valid, bins=60, color="#888888", edgecolor="none",
                 density=True, orientation="vertical")
    ax_hist.axvline(med_v, color="#333333", lw=1.2, ls=":", zorder=2,
                    label=f"median")
    for thresh in THRESHOLDS:
        is_chosen = (thresh == CHOSEN)
        nsig = (thresh - med_v) / std_v
        ax_hist.axvline(thresh,
                        color=HIGHLIGHT_COLOR if is_chosen else "#444444",
                        lw=2.2 if is_chosen else 1.2,
                        ls="-"  if is_chosen else "--",
                        zorder=3 if is_chosen else 2,
                        label=f"{thresh:.2f} m  ({nsig:+.1f}σ)"
                              + (" ✓" if is_chosen else ""))
    # Secondary x-axis on top showing σ from median
    ax_top = ax_hist.secondary_xaxis(
        "top",
        functions=(lambda x: (x - med_v) / std_v,
                   lambda z: z * std_v + med_v),
    )
    ax_top.set_xlabel("σ from median", fontsize=FS - 4)
    ax_top.tick_params(labelsize=FS - 4)
    ax_hist.set_xlabel("Amplitude  [m]", fontsize=FS - 3)
    ax_hist.set_ylabel("Density", fontsize=FS - 3)
    ax_hist.set_title("Value distribution", fontsize=FS_TITLE - 1, pad=6)
    ax_hist.legend(fontsize=FS - 4, title="Threshold  (σ from median)",
                   title_fontsize=FS - 5, frameon=True)
    ax_hist.tick_params(labelsize=FS - 3)

    # ---- Threshold comparison panels ----------------------------------------
    cmap_bin = ListedColormap([NAN_COLOR, "white", "#d62728"])
    norm_bin = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    for ax, thresh in zip(ax_t, THRESHOLDS):
        binary                  = np.full_like(values, np.nan)
        binary[has_data]        = np.where(values[has_data] > thresh, 1.0, 0.0)
        ax.imshow(
            np.ma.masked_invalid(binary),
            cmap=cmap_bin, norm=norm_bin,
            origin="lower", extent=extent,
            interpolation="nearest",
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {thresh:.2f} m", fontsize=FS_TITLE - 1, pad=4)
        if thresh == CHOSEN:
            for spine in ax.spines.values():
                spine.set_edgecolor(HIGHLIGHT_COLOR)
                spine.set_linewidth(3.5)
        frac = float(np.mean(binary[has_data] == 1.0)) * 100
        ax.text(0.97, 0.03, f"{frac:.1f} % flagged",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS - 4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.85))

    fig.suptitle(
        f"Amplitude-based dynamic-seabed cost map  —  chosen threshold = {CHOSEN} m",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 26 — Combined cost-map overview (bathymetry + all 3 overlays)
# ---------------------------------------------------------------------------

def plot_combined_costmaps(save_path=None):
    """
    Bathymetry base (all cells combined, cmocean.deep) with three
    semi-transparent risk overlays and both proposed routes.

      Red  : sandwave detection probability (continuous alpha, prob = 1 → α 0.65)
      Blue : high-variance cells  (log-std > 0.05, binary, α 0.55)
      Gold : high-amplitude cells (amplitude > 0.08 m,   binary, α 0.60)

    Where layers overlap the RGBA compositing naturally mixes the colours,
    giving a visual indication of how many risk factors coincide.
    """
    import skimage as ski
    import costmap as cm_mod
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    LABELS_DIR = "sandwave_detection_v8/labels"
    VAR_DIR    = "variance_rasters/Rasters"
    AMP_DIR    = "amplitude_rasters/Rasters_amp"
    VAR_THRESH = 0.05
    AMP_THRESH = 0.08
    SW_ALPHA   = 0.65
    VAR_ALPHA  = 0.55
    AMP_ALPHA  = 0.60

    def _load_cells(directory, transform_fn=None, combine_fn=None):
        """Scan *directory*, group by cell id, combine, optional-transform, push into CostMap."""
        if combine_fn is None:
            combine_fn = np.nanmean
        cmap_out = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
        d: dict = {}
        for fname in os.listdir(directory):
            cid = fname.split("_")[1]
            arr = np.load(os.path.join(directory, fname))[::-1, :]
            d.setdefault(cid, []).append(arr)
        for cid, rasters in d.items():
            with np.errstate(invalid="ignore"):
                avg = combine_fn(np.stack(rasters, axis=0), axis=0)
            xs, xe, ys, ye = cmap_out.slice_cost_map(int(cid))
            if transform_fn is not None:
                avg = transform_fn(avg)
            cmap_out.add_cost(xs, ys, cost=avg, x_idx_end=xe, y_idx_end=ye)
        return cmap_out

    # ---- Bathymetry ---------------------------------------------------------
    print("  Loading bathymetry …", flush=True)
    bath = _load_cells(
        "destriped_rasters",
        transform_fn=lambda a: ski.measure.block_reduce(a, block_size=5,
                                                        func=np.nanmean),
    )

    # ---- Sandwave probability -----------------------------------------------
    print("  Loading sandwave labels …", flush=True)
    sw_raw = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    sw_d: dict = {}
    for fname in [f for f in os.listdir(LABELS_DIR)
                  if f.endswith("destriped_labels_smoothed.npy")]:
        cid = fname.split("_")[1]
        arr = np.load(os.path.join(LABELS_DIR, fname))[::-1, :].astype(float)
        arr[arr == -1] = np.nan
        sw_d.setdefault(cid, []).append(arr)
    for cid, rasters in sw_d.items():
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(np.stack(rasters, axis=0), axis=0)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.max)
        xs, xe, ys, ye = sw_raw.slice_cost_map(int(cid))
        sw_raw.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    # ---- Variance (log-std) -------------------------------------------------
    print("  Loading variance …", flush=True)
    var_c = _load_cells(
        VAR_DIR,
        transform_fn=lambda a: np.log10(
            np.sqrt(np.maximum(
                ski.measure.block_reduce(a, block_size=5, func=np.nanmean), 0.0
            )) + 1.0
        ),
    )

    # ---- Amplitude ----------------------------------------------------------
    print("  Loading amplitude …", flush=True)
    amp_c = _load_cells(
        AMP_DIR,
        transform_fn=lambda a: ski.measure.block_reduce(a, block_size=5,
                                                        func=np.nanmax),
        combine_fn=np.nanmax,
    )

    # ---- Extent -------------------------------------------------------------
    bl     = bath.bl
    extent = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]

    # ---- Build RGBA overlays ------------------------------------------------
    shape = bath.costs.shape

    # Sandwave: red, alpha ∝ probability
    sw_prob              = np.clip(np.where(np.isnan(sw_raw.costs), 0.0,
                                            sw_raw.costs), 0.0, 1.0)
    sw_rgba              = np.zeros((*shape, 4), dtype=float)
    sw_rgba[..., 0]      = 0.84
    sw_rgba[..., 1]      = 0.15
    sw_rgba[..., 2]      = 0.16
    sw_rgba[..., 3]      = sw_prob * SW_ALPHA

    # Variance: vivid magenta (distinct from background blues/greens), binary
    var_flag             = (~np.isnan(var_c.costs)) & (var_c.costs > VAR_THRESH)
    var_rgba             = np.zeros((*shape, 4), dtype=float)
    var_rgba[var_flag, 0] = 0.80
    var_rgba[var_flag, 1] = 0.10
    var_rgba[var_flag, 2] = 0.80
    var_rgba[var_flag, 3] = VAR_ALPHA

    # Amplitude: gold, binary
    amp_flag             = (~np.isnan(amp_c.costs)) & (amp_c.costs > AMP_THRESH)
    amp_rgba             = np.zeros((*shape, 4), dtype=float)
    amp_rgba[amp_flag, 0] = 1.00
    amp_rgba[amp_flag, 1] = 0.75
    amp_rgba[amp_flag, 2] = 0.00
    amp_rgba[amp_flag, 3] = AMP_ALPHA

    # ---- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7.5), constrained_layout=True)

    cm_bath2 = copy.copy(CMAP_BATH)
    cm_bath2.set_bad(NAN_COLOR)
    im = ax.imshow(np.ma.masked_invalid(bath.costs),
                   cmap=cm_bath2, origin="lower", extent=extent,
                   interpolation="nearest", aspect="equal", zorder=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Depth  [m]", fontsize=FS - 2)
    cbar.ax.tick_params(labelsize=FS - 3)

    for rgba in (sw_rgba, var_rgba, amp_rgba):
        ax.imshow(rgba, origin="lower", extent=extent,
                  interpolation="nearest", zorder=2)

    # ---- Routes -------------------------------------------------------------
    ROUTE_FILES  = ["shapes/line1.shp", "shapes/line2.shp"]
    ROUTE_COLORS = ["#aeea00", "#ff7f0e"]   # chartreuse + orange: both pop on blue-green
    ROUTE_LABELS = ["Route 1", "Route 2"]
    try:
        import geopandas as gpd
        for fpath, color, label in zip(ROUTE_FILES, ROUTE_COLORS, ROUTE_LABELS):
            if os.path.exists(fpath):
                gpd.read_file(fpath).to_crs(bath.csr).plot(
                    ax=ax, color=color, linewidth=2.0, label=label, zorder=5)
    except ImportError:
        for route, color, label in zip(bath.routes, ROUTE_COLORS, ROUTE_LABELS):
            ys_r, xs_r = np.where(route)
            ax.scatter(bl[0] + xs_r * 100 + 50, bl[1] + ys_r * 100 + 50,
                       c=color, s=2, zorder=5, label=label, linewidths=0)

    # ---- Axes ---------------------------------------------------------------
    ax.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS)
    ax.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=6)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 2)
    plt.setp(ax.get_yticklabels(), fontsize=FS - 2)
    ax.set_title("Combined dynamic-risk cost maps",
                 fontsize=FS_TITLE + 1, fontweight="bold", pad=10)

    # ---- Legend -------------------------------------------------------------
    sw_patch  = Patch(facecolor=(0.84, 0.15, 0.16, SW_ALPHA),
                      edgecolor="none", label="Sandwave detection  (prob. weighted)")
    var_patch = Patch(facecolor=(0.80, 0.10, 0.80, VAR_ALPHA),
                      edgecolor="none",
                      label=f"High variance  (log-std > {VAR_THRESH})")
    amp_patch = Patch(facecolor=(1.00, 0.75, 0.00, AMP_ALPHA),
                      edgecolor="none",
                      label=f"High amplitude  (> {AMP_THRESH} m)")
    no_data   = Patch(facecolor=NAN_COLOR, edgecolor="none", label="No survey data")
    route_handles = [
        Line2D([0], [0], color=c, lw=2, label=l)
        for c, l in zip(ROUTE_COLORS, ROUTE_LABELS)
    ]
    ax.legend(handles=[sw_patch, var_patch, amp_patch, no_data, *route_handles],
              loc="upper left", fontsize=FS - 2, frameon=True, edgecolor="#cccccc")

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 25 — Cell 56 local-std masking pipeline
# ---------------------------------------------------------------------------

def plot_local_std_mask_cell(cell, cdi_a, cdi_b, save_path=None):
    """
    Six-panel illustration of the local-std masking pipeline for a given cell.

    Row 0:  (a) Survey 1 bathymetry
            (b) Survey 2 bathymetry          [shared depth scale with (a)]
            (c) Change map: Survey 2 − Survey 1 (diverging colormap)

    Row 1:  (d) Local std of change (√variance), full 250×250 resolution
            (e) Log-std after 5× downscale to 50×50; threshold line at 0.05
            (f) Dynamic-area mask (threshold = 0.05) overlaid on bathymetry
    """
    import skimage as ski
    from matplotlib.colors import TwoSlopeNorm

    CELL        = cell
    CDI_A       = cdi_a   # Survey 1
    CDI_B       = cdi_b   # Survey 2
    VAR_FILE    = f"variance_rasters/Rasters/cell_{CELL}_local_var_diff_{CDI_B}_{CDI_A}.npy"
    BATH_A_FILE = f"destriped_rasters/cell_{CELL}_CDI_{CDI_A}_destriped.npy"
    BATH_B_FILE = f"destriped_rasters/cell_{CELL}_CDI_{CDI_B}_destriped.npy"
    VAR_THRESH  = 0.05

    # ---- Load data ----------------------------------------------------------
    r1  = np.load(BATH_A_FILE)           # 250×250, depths in m
    r2  = np.load(BATH_B_FILE)
    var = np.load(VAR_FILE)              # local variance of (r2-r1), 250×250

    diff = r2 - r1                       # change map
    std_full = np.sqrt(np.maximum(var, 0.0))   # local std, 250×250

    # Downscale & log-transform (replicates build_variance_costmap pipeline)
    rescaled    = ski.measure.block_reduce(var, block_size=5, func=np.nanmean)  # 50×50
    log_std     = np.log10(np.sqrt(np.maximum(rescaled, 0.0)) + 1.0)           # 50×50
    mask_binary = (log_std > VAR_THRESH).astype(float)

    # Average bathymetry for overlay panel, also downscaled to 50×50
    bath_avg    = np.nanmean(np.stack([r1, r2], axis=0), axis=0)
    bath_ds     = ski.measure.block_reduce(bath_avg, block_size=5, func=np.nanmean)

    nan_mask_full = np.isnan(r1) | np.isnan(r2)
    nan_mask_ds   = ski.measure.block_reduce(
        nan_mask_full.astype(float), block_size=5, func=np.max).astype(bool)

    # ---- Shared colour limits -----------------------------------------------
    vmin_b = float(np.nanpercentile(np.stack([r1, r2]), 2))
    vmax_b = float(np.nanpercentile(np.stack([r1, r2]), 98))
    diff_abs = float(np.nanpercentile(np.abs(diff), 98))
    std_vmax = float(np.nanpercentile(std_full, 99))
    log_vmax = float(np.nanpercentile(log_std,  99))

    cm_bath2 = copy.copy(CMAP_BATH); cm_bath2.set_bad(NAN_COLOR)
    cm_std   = copy.copy(plt.colormaps["YlOrRd"]); cm_std.set_bad(NAN_COLOR)

    # ---- Layout -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             constrained_layout=True)
    ax_b1, ax_b2, ax_ch = axes[0]
    ax_sd, ax_ls, ax_mk = axes[1]

    def _off(ax):
        ax.set_xticks([]); ax.set_yticks([])

    # ---- (a) Survey 1 -------------------------------------------------------
    im_b = ax_b1.imshow(np.ma.masked_where(nan_mask_full, r1),
                        cmap=cm_bath2, vmin=vmin_b, vmax=vmax_b, origin="upper")
    _off(ax_b1)
    ax_b1.set_title(f"(a) Survey 1  (CDI {CDI_A})", fontsize=FS_TITLE, pad=5)
    fig.colorbar(im_b, ax=ax_b1, shrink=0.85, label="Depth  [m]")

    # ---- (b) Survey 2 -------------------------------------------------------
    ax_b2.imshow(np.ma.masked_where(nan_mask_full, r2),
                 cmap=cm_bath2, vmin=vmin_b, vmax=vmax_b, origin="upper")
    _off(ax_b2)
    ax_b2.set_title(f"(b) Survey 2  (CDI {CDI_B})", fontsize=FS_TITLE, pad=5)
    fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin_b, vmax_b), cmap=cm_bath2),
        ax=ax_b2, shrink=0.85, label="Depth  [m]")

    # ---- (c) Change map -----------------------------------------------------
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs)
    im_ch = ax_ch.imshow(np.ma.masked_where(nan_mask_full, diff),
                         cmap="RdBu_r", norm=diff_norm, origin="upper")
    _off(ax_ch)
    ax_ch.set_title("(c) Change: Survey 2 − Survey 1", fontsize=FS_TITLE, pad=5)
    fig.colorbar(im_ch, ax=ax_ch, shrink=0.85, label="Δdepth  [m]")

    # ---- (d) Local std of change (full resolution) --------------------------
    im_sd = ax_sd.imshow(np.ma.masked_where(nan_mask_full, std_full),
                         cmap=cm_std, vmin=0, vmax=std_vmax, origin="upper")
    _off(ax_sd)
    ax_sd.set_title("(d) Local std of change  (250×250)", fontsize=FS_TITLE, pad=5)
    fig.colorbar(im_sd, ax=ax_sd, shrink=0.85, label="Std  [m]")

    # ---- (e) Log-std after 5× downscale, threshold annotated ---------------
    im_ls = ax_ls.imshow(np.ma.masked_where(nan_mask_ds, log_std),
                         cmap=cm_std, vmin=0, vmax=log_vmax, origin="upper")
    _off(ax_ls)
    ax_ls.set_title(r"(e) $\log_{10}(\mathrm{std}+1)$ after 5× downscale  (50×50)",
                    fontsize=FS_TITLE, pad=5)
    cb_ls = fig.colorbar(im_ls, ax=ax_ls, shrink=0.85,
                         label=r"$\log_{10}(\sqrt{\sigma^2}+1)$  [–]")
    # Mark threshold on colorbar
    cb_ls.ax.axhline(y=VAR_THRESH / log_vmax, color=HIGHLIGHT_COLOR, lw=2.0)
    cb_ls.ax.text(0.5, VAR_THRESH / log_vmax + 0.03, f"t = {VAR_THRESH}",
                  transform=cb_ls.ax.transAxes, ha="center", va="bottom",
                  fontsize=FS - 4, color=HIGHLIGHT_COLOR, fontweight="bold")
    frac = float(np.mean(mask_binary[~nan_mask_ds])) * 100
    ax_ls.text(0.97, 0.03, f"{frac:.0f} % flagged",
               transform=ax_ls.transAxes, ha="right", va="bottom",
               fontsize=FS - 3,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="none", alpha=0.85))

    # ---- (f) Binary mask overlaid on downscaled bathymetry ------------------
    ax_mk.imshow(np.ma.masked_where(nan_mask_ds, bath_ds),
                 cmap=cm_bath2, vmin=vmin_b, vmax=vmax_b, origin="upper")
    # Red overlay where flagged
    red_overlay        = np.zeros((*mask_binary.shape, 4), dtype=float)
    red_overlay[..., 0] = 0.84
    red_overlay[..., 1] = 0.15
    red_overlay[..., 2] = 0.16
    red_overlay[..., 3] = np.where(~nan_mask_ds & (mask_binary == 1), 0.65, 0.0)
    ax_mk.imshow(red_overlay, origin="upper")
    _off(ax_mk)
    ax_mk.set_title(f"(f) Dynamic-area mask  (t = {VAR_THRESH})", fontsize=FS_TITLE, pad=5)
    from matplotlib.patches import Patch
    ax_mk.legend(
        handles=[Patch(facecolor=(0.84, 0.15, 0.16, 0.65), label="Flagged (dynamic)"),
                 Patch(facecolor=NAN_COLOR, label="No data")],
        loc="lower right", fontsize=FS - 4, frameon=True, edgecolor="#cccccc")

    fig.suptitle(
        f"Cell {CELL}: local-std masking pipeline  "
        f"(surveys {CDI_A} & {CDI_B}, threshold = {VAR_THRESH})",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    _save_or_show(fig, save_path)


def plot_local_std_mask_cell56(save_path=None):
    plot_local_std_mask_cell(56, cdi_a=2174760, cdi_b=3844672, save_path=save_path)


def plot_local_std_mask_cell39(save_path=None):
    plot_local_std_mask_cell(39, cdi_a=2174760, cdi_b=3844672, save_path=save_path)


# ---------------------------------------------------------------------------
# Figure 27 — A* optimised route: unrestricted vs N2000 exclusion
# ---------------------------------------------------------------------------

def plot_optimised_route(save_path=None, fill_nn=False):
    """
    Two-panel (stacked) comparison of the A*-optimised pipeline route:
      Top    : Optimised route on the combined risk cost map (no exclusion zones).
      Bottom : Same cost map but with N2000 areas blocked (cost = -1, impassable).

    Background    : Combined destriped bathymetry (cmocean.deep).
    Risk overlays : Same three layers as Fig 26 (sandwave / variance / amplitude).
    N2000 overlay : Gray semi-transparent mask in the bottom panel only.
    A* path       : White line with dark outline for maximum contrast.
    Ref routes    : Chartreuse (Route 1) and orange (Route 2) for comparison.

    Routing parameters match test_optimiser.run():
      momentum=8, max_turn_steps=1, heuristic_weight=1.0, rescale_factor=1.
    """
    import skimage as ski
    import costmap as cm_mod
    from Astar import AStarPlanner
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke

    LABELS_DIR = "sandwave_detection_v8/labels"
    VAR_DIR    = "variance_rasters/Rasters"
    AMP_DIR    = "amplitude_rasters/Rasters_amp"
    N2K_SHP    = "shapes/n200.shp"

    VAR_THRESH = 0.05
    AMP_THRESH = 0.08
    SW_ALPHA   = 0.65
    VAR_ALPHA  = 0.55
    AMP_ALPHA  = 0.60
    N2K_ALPHA  = 0.72

    ROUTE_FILES  = ["shapes/line1.shp", "shapes/line2.shp"]
    ROUTE_COLORS = ["#aeea00", "#ff7f0e"]   # chartreuse + orange (same as Fig 26)
    ROUTE_LABELS = ["Route 1 (proposed)", "Route 2 (proposed)"]
    PATH_COLOR   = "white"
    PATH_EDGE    = "#111111"

    # ---- Helpers ----------------------------------------------------------------
    def _load_dir(directory, suffix=None):
        d: dict = {}
        for fname in os.listdir(directory):
            if suffix and not fname.endswith(suffix):
                continue
            cid = fname.split("_")[1]
            arr = np.load(os.path.join(directory, fname))[::-1, :]
            d.setdefault(cid, []).append(arr)
        return d

    def _combine(rasters, func=np.nanmean):
        stacked = np.stack(rasters, axis=0)
        with np.errstate(invalid="ignore"):
            return func(stacked, axis=0)

    # ---- Bathymetry (also serves as the nan-mask source) -------------------------
    print("  Building bathymetry …", flush=True)
    bath = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    raster_d = _load_dir("destriped_rasters")
    for cid, rasters in raster_d.items():
        avg      = _combine(rasters, np.nanmean)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        xs, xe, ys, ye = bath.slice_cost_map(int(cid))
        bath.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    # ---- Sandwave probability (0–1, continuous) ----------------------------------
    print("  Loading sandwave labels …", flush=True)
    sw_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(LABELS_DIR,
                                   suffix="destriped_labels_smoothed.npy").items():
        rasters = [r.astype(float) for r in rasters]
        for r in rasters:
            r[r == -1] = np.nan
        avg      = _combine(rasters, np.nanmean)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.max)
        xs, xe, ys, ye = sw_cm.slice_cost_map(int(cid))
        sw_cm.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    # ---- Variance (log-std, raw; binarised for routing) -------------------------
    print("  Loading variance …", flush=True)
    var_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(VAR_DIR).items():
        avg         = _combine(rasters, np.nanmean)
        rescaled    = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        transformed = np.log10(np.sqrt(np.maximum(rescaled, 0.0)) + 1.0)
        xs, xe, ys, ye = var_cm.slice_cost_map(int(cid))
        var_cm.add_cost(xs, ys, cost=transformed, x_idx_end=xe, y_idx_end=ye)

    # Binarised copy for routing cost (1 where above threshold, NaN elsewhere)
    var_bin = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    var_bin.set_cost(None, None,
                     cost=np.where((~np.isnan(var_cm.costs)) &
                                   (var_cm.costs > VAR_THRESH), 1.0, np.nan))

    # ---- Amplitude (raw; binarised for routing) ----------------------------------
    print("  Loading amplitude …", flush=True)
    amp_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(AMP_DIR).items():
        avg      = _combine(rasters, np.nanmax)
        rescaled = ski.measure.block_reduce(avg, block_size=5, func=np.nanmax)
        xs, xe, ys, ye = amp_cm.slice_cost_map(int(cid))
        amp_cm.add_cost(xs, ys, cost=rescaled, x_idx_end=xe, y_idx_end=ye)

    amp_bin = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    amp_bin.set_cost(None, None,
                     cost=np.where((~np.isnan(amp_cm.costs)) &
                                   (amp_cm.costs > AMP_THRESH), 1.0, np.nan))

    # ---- Build routing cost grids (matching test_optimiser.run()) ---------------
    def _routing_grid(with_n2000: bool) -> np.ndarray:
        """Replicate the cost-map assembly in test_optimiser.run()."""
        costmap = cm_mod.CostMap(dx=100, dy=100, default_cost=1)
        costmap.add_cost(None, None, cost=sw_cm.costs)   # continuous sandwave prob
        costmap.add_cost(None, None, cost=var_bin.costs)  # binary variance flag
        costmap.add_cost(None, None, cost=amp_bin.costs)  # binary amplitude flag
        costmap.set_nans(bath)                            # NaN where no bathymetry
        if with_n2000:
            try:
                costmap.block_n2000(path=N2K_SHP)        # -1 (impassable) in N2000
            except Exception as exc:
                print(f"    [warn] block_n2000 failed: {exc}", flush=True)
        if fill_nn:
            costmap.fill_nans_nn(max_gap=1000)            # small gaps → nearest-neighbour
            costmap.fill_nans_high_cost()                 # remaining large gaps → max cost
        else:
            costmap.fill_nans_high_cost()                 # NaN → max finite cost
        return costmap.costs

    print("  Building routing cost grids …", flush=True)
    grid_free = _routing_grid(with_n2000=False)
    grid_n2k  = _routing_grid(with_n2000=True)

    # ---- N2000 mask for the visualisation overlay --------------------------------
    n2k_mask = np.zeros(bath.costs.shape, dtype=bool)
    try:
        temp = cm_mod.CostMap(dx=100, dy=100, default_cost=0.0)
        arr  = np.zeros(bath.costs.shape, dtype=float)
        temp.block_n2000(array=arr, path=N2K_SHP)
        n2k_mask = (arr == -1)
    except Exception as exc:
        print(f"  [warn] N2000 mask not available: {exc}", flush=True)

    # ---- Start / goal indices (match test_optimiser.run()) ----------------------
    ref_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=1)

    # Route cost helper (needs ref_cm.routes and both grids)
    route_masks = ref_cm.routes   # list of bool arrays, one per shapefile
    grids = {"free": grid_free, "n2k": grid_n2k}

    def _route_cost(grid, mask):
        """Sum cell costs along a boolean route mask, ignoring no-go cells."""
        valid = mask & np.isfinite(grid) & (grid > 0)
        return float(grid[valid].sum())

    sx, sy = ref_cm.start_utm
    ex, ey = ref_cm.end_utm
    sx_idx, sy_idx = ref_cm.get_idx_from_coordinates(sx, sy)
    ex_idx, ey_idx = ref_cm.get_idx_from_coordinates(ex, ey)

    # ---- Run A* for both scenarios ----------------------------------------------
    paths: dict = {}
    for label, grid in [("free", grid_free), ("n2k", grid_n2k)]:
        print(f"  Running A* ({label}) …", flush=True)
        planner = AStarPlanner(
            cost_grid=grid,
            max_turn_steps=1,
            heuristic_weight=1.0,
            momentum=8,
        )
        try:
            result = planner.solve(
                start=(sy_idx, sx_idx),
                goal=(ey_idx, ex_idx),
                start_heading=None,
                goal_heading=None,
            )
            paths[label] = result
            if result:
                print(f"    -> {len(result.coords)} steps, "
                      f"cost {result.total_cost:.1f}", flush=True)
            else:
                print("    -> no path found", flush=True)
        except Exception as exc:
            print(f"    -> A* failed: {exc}", flush=True)
            paths[label] = None

    # ---- Optional NN fill of bathymetry NaNs (for display only) ----------------
    # Routing grids are already built so modifying bath.costs here is safe.
    if fill_nn:
        print("  Filling bathymetry NaNs (nearest-neighbour, max_gap=1000) …", flush=True)
        bath.fill_nans_nn(max_gap=1000)

    # ---- RGBA overlays (same palette as Fig 26) ---------------------------------
    bl     = bath.bl
    extent = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]
    shape  = bath.costs.shape

    sw_prob         = np.clip(np.where(np.isnan(sw_cm.costs), 0.0, sw_cm.costs),
                              0.0, 1.0)
    sw_rgba         = np.zeros((*shape, 4), dtype=float)
    sw_rgba[..., 0] = 0.84; sw_rgba[..., 1] = 0.15; sw_rgba[..., 2] = 0.16
    sw_rgba[..., 3] = sw_prob * SW_ALPHA

    var_flag          = (~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH)
    var_rgba          = np.zeros((*shape, 4), dtype=float)
    var_rgba[var_flag, 0] = 0.80; var_rgba[var_flag, 1] = 0.10
    var_rgba[var_flag, 2] = 0.80; var_rgba[var_flag, 3] = VAR_ALPHA

    amp_flag          = (~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH)
    amp_rgba          = np.zeros((*shape, 4), dtype=float)
    amp_rgba[amp_flag, 0] = 1.00; amp_rgba[amp_flag, 1] = 0.75
    amp_rgba[amp_flag, 2] = 0.00; amp_rgba[amp_flag, 3] = AMP_ALPHA

    n2k_rgba          = np.zeros((*shape, 4), dtype=float)
    n2k_rgba[n2k_mask, 0] = 0.50; n2k_rgba[n2k_mask, 1] = 0.50
    n2k_rgba[n2k_mask, 2] = 0.50; n2k_rgba[n2k_mask, 3] = N2K_ALPHA

    # ---- Helper: A* path (row, col) → UTM (x, y) --------------------------------
    def _path_utm(result):
        if result is None:
            return None, None
        xs = [bl[0] + col * 100 + 50 for (row, col) in result.coords]
        ys = [bl[1] + row * 100 + 50 for (row, col) in result.coords]
        return xs, ys

    # ---- Figure (2 rows × 1 col) ------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 13), constrained_layout=True)

    cm_bath2 = copy.copy(CMAP_BATH)
    cm_bath2.set_bad(NAN_COLOR)

    configs = [
        ("free", "Optimised route — unrestricted",                False),
        ("n2k",  "Optimised route — N2000 zones excluded",        True),
    ]
    im_ref = None

    for ax, (key, title, show_n2k) in zip(axes, configs):
        # Bathymetry base (NN-filled → no masking needed; else mask NaNs)
        bath_data = bath.costs if fill_nn else np.ma.masked_invalid(bath.costs)
        im = ax.imshow(
            bath_data,
            cmap=cm_bath2, origin="lower", extent=extent,
            interpolation="nearest", aspect="equal", zorder=1,
        )
        if im_ref is None:
            im_ref = im

        # Risk overlays
        for rgba in (sw_rgba, var_rgba, amp_rgba):
            ax.imshow(rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=2)

        # N2000 overlay (bottom panel only)
        if show_n2k and n2k_mask.any():
            ax.imshow(n2k_rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=3)

        # Reference routes (vector lines via geopandas)
        try:
            import geopandas as gpd
            for fpath, color, rlabel in zip(ROUTE_FILES, ROUTE_COLORS, ROUTE_LABELS):
                if os.path.exists(fpath):
                    gpd.read_file(fpath).to_crs(bath.csr).plot(
                        ax=ax, color=color, linewidth=2.0, label=rlabel, zorder=5)
        except ImportError:
            pass

        # A* path — white with dark outline via path_effects
        xs_p, ys_p = _path_utm(paths[key])
        if xs_p is not None:
            ax.plot(
                xs_p, ys_p,
                color=PATH_COLOR, lw=2.2,
                solid_capstyle="round", solid_joinstyle="round",
                path_effects=[withStroke(linewidth=4.5, foreground=PATH_EDGE)],
                zorder=7,
            )

        # ---- Legend (per panel, with costs) ------------------------------------
        sw_patch  = Patch(facecolor=(0.84, 0.15, 0.16, SW_ALPHA), edgecolor="none",
                          label="Sandwave detection  (prob. weighted)")
        var_patch = Patch(facecolor=(0.80, 0.10, 0.80, VAR_ALPHA), edgecolor="none",
                          label=f"High variance  (log-std > {VAR_THRESH})")
        amp_patch = Patch(facecolor=(1.00, 0.75, 0.00, AMP_ALPHA), edgecolor="none",
                          label=f"High amplitude  (> {AMP_THRESH} m)")
        n2k_patch = Patch(facecolor=(0.50, 0.50, 0.50, N2K_ALPHA), edgecolor="none",
                          label="N2000 exclusion zone  (impassable)")

        astar_cost = paths[key].total_cost if paths[key] else float("nan")
        path_line_h = Line2D(
            [0], [0], color=PATH_COLOR, lw=2.2,
            path_effects=[withStroke(linewidth=4.5, foreground=PATH_EDGE)],
            label=f"A* optimal route  (cost: {astar_cost:.0f})",
        )
        grid_for_cost = grids[key]
        route_hdl = [
            Line2D([0], [0], color=c, lw=2,
                   label=f"{l}  (cost: {_route_cost(grid_for_cost, mask):.0f})")
            for c, l, mask in zip(ROUTE_COLORS, ROUTE_LABELS, route_masks)
        ]

        leg_handles = [path_line_h, *route_hdl, sw_patch, var_patch, amp_patch]
        if not fill_nn:
            no_data = Patch(facecolor=NAN_COLOR, edgecolor="none", label="No survey data")
            leg_handles.append(no_data)
        if show_n2k:
            leg_handles.insert(1, n2k_patch)
        ax.legend(handles=leg_handles, loc="upper left",
                  fontsize=FS - 4, frameon=True, edgecolor="#cccccc")

        # Axes
        ax.set_title(title, fontsize=FS_TITLE, pad=7, fontweight="bold")
        ax.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS - 2)
        ax.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS - 2)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 3)
        plt.setp(ax.get_yticklabels(), fontsize=FS - 3)

    # Shared colorbar on the right
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes.tolist(), shrink=0.75, pad=0.02)
        cbar.set_label("Depth  [m]", fontsize=FS - 2)
        cbar.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        "A* optimised pipeline route  "
        "(momentum = 8,  max turn = 45°,  100 m cell resolution)",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    _save_or_show(fig, save_path)


def plot_optimised_route_nn(save_path=None):
    """Like plot_optimised_route but with NaN bathymetry cells filled by nearest-neighbour."""
    plot_optimised_route(save_path=save_path, fill_nn=True)


# ---------------------------------------------------------------------------
# Figures 28/28b — Sensitivity analysis: route frequency heatmap + consensus
# ---------------------------------------------------------------------------

def plot_sensitivity_route(save_path=None, fill_nn=False):
    """
    Two-panel sensitivity analysis of the A* pipeline route.

    Runs A* n=100 times with perturbed cost maps (Gaussian noise σ=0.1),
    builds a route-frequency heatmap, then finds the consensus route via A*
    on an inverse-frequency cost map.

    Parameters match run_sensitivity_analysis() in test_optimiser.py:
      n=100, sigma=0.1, rescale_factor=4, momentum=2.

    Top panel   : Unrestricted routing.
    Bottom panel: N2000 zones blocked (cost = -1, impassable).
    """
    import skimage as ski
    import costmap as cm_mod
    from Astar import AStarPlanner
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke

    # ---- Parameters ------------------------------------------------------------
    N          = 100
    SIGMA      = 0.5
    VAR_THRESH = 0.05
    AMP_THRESH = 0.08
    RESCALE    = 4
    MOMENTUM   = 2

    LABELS_DIR = "sandwave_detection_v8/labels"
    VAR_DIR    = "variance_rasters/Rasters"
    AMP_DIR    = "amplitude_rasters/Rasters_amp"
    N2K_SHP    = "shapes/n200.shp"

    SW_ALPHA   = 0.45
    VAR_ALPHA  = 0.40
    AMP_ALPHA  = 0.45
    N2K_ALPHA  = 0.60
    HEAT_ALPHA = 0.70

    ROUTE_FILES  = ["shapes/line1.shp", "shapes/line2.shp"]
    ROUTE_COLORS = ["#aeea00", "#ff7f0e"]
    ROUTE_LABELS = ["Route 1 (proposed)", "Route 2 (proposed)"]
    PATH_COLOR   = "white"
    PATH_EDGE    = "#111111"

    # ---- Helpers ---------------------------------------------------------------
    def _load_dir(directory, suffix=None):
        d = {}
        for fname in os.listdir(directory):
            if suffix and not fname.endswith(suffix):
                continue
            cid = fname.split("_")[1]
            arr = np.load(os.path.join(directory, fname))[::-1, :]
            d.setdefault(cid, []).append(arr)
        return d

    def _combine(rasters, func=np.nanmean):
        stacked = np.stack(rasters, axis=0)
        with np.errstate(invalid="ignore"):
            return func(stacked, axis=0)

    def _add_noise(arr, sigma):
        noisy = arr + np.random.normal(0, sigma, arr.shape)
        return np.where((~np.isnan(arr)) & (arr != 0), noisy, arr)

    # ---- Load data layers (full resolution 800×350) ----------------------------
    print("  Building bathymetry …", flush=True)
    bath = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir("destriped_rasters").items():
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        xs, xe, ys, ye = bath.slice_cost_map(int(cid))
        bath.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    print("  Loading sandwave labels …", flush=True)
    sw_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(LABELS_DIR,
                                   suffix="destriped_labels_smoothed.npy").items():
        rasters = [r.astype(float) for r in rasters]
        for r in rasters:
            r[r == -1] = np.nan
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.max)
        xs, xe, ys, ye = sw_cm.slice_cost_map(int(cid))
        sw_cm.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    print("  Loading variance …", flush=True)
    var_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(VAR_DIR).items():
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        ts  = np.log10(np.sqrt(np.maximum(rs, 0.0)) + 1.0)
        xs, xe, ys, ye = var_cm.slice_cost_map(int(cid))
        var_cm.add_cost(xs, ys, cost=ts, x_idx_end=xe, y_idx_end=ye)

    print("  Loading amplitude …", flush=True)
    amp_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(AMP_DIR).items():
        avg = _combine(rasters, np.nanmax)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmax)
        xs, xe, ys, ye = amp_cm.slice_cost_map(int(cid))
        amp_cm.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    # ---- Base arrays for sensitivity (fixed across iterations) -----------------
    full_shape      = bath.costs.shape
    nan_mask_full   = np.isnan(bath.costs)
    sw_base         = sw_cm.costs.copy()
    var_base_thresh = np.where(
        (~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH), 1.0, np.nan)
    amp_base_thresh = np.where(
        (~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH), 1.0, np.nan)

    # ---- N2000 mask (boolean, full resolution) ----------------------------------
    n2k_mask = np.zeros(full_shape, dtype=bool)
    try:
        arr = np.zeros(full_shape, dtype=float)
        cm_mod.CostMap(dx=100, dy=100, default_cost=0.0).block_n2000(
            array=arr, path=N2K_SHP)
        n2k_mask = (arr == -1)
    except Exception as exc:
        print(f"  [warn] N2000 mask: {exc}", flush=True)

    # ---- Start / goal indices ---------------------------------------------------
    ref_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=1)
    sx, sy = ref_cm.start_utm
    ex, ey = ref_cm.end_utm
    sx_idx, sy_idx = ref_cm.get_idx_from_coordinates(sx, sy)
    ex_idx, ey_idx = ref_cm.get_idx_from_coordinates(ex, ey)
    sx_rs = sx_idx // RESCALE
    sy_rs = sy_idx // RESCALE
    ex_rs = ex_idx // RESCALE
    ey_rs = ey_idx // RESCALE

    # Rescaled masks
    nan_mask_rs = ski.measure.block_reduce(
        nan_mask_full.astype(float), block_size=RESCALE, func=np.max).astype(bool)
    n2k_mask_rs = ski.measure.block_reduce(
        n2k_mask.astype(float), block_size=RESCALE, func=np.max).astype(bool)

    # ---- Precompute NN fill info (only needed when fill_nn=True) ---------------
    # Building the KD-tree once avoids rebuilding it 2×N times in the loop.
    fill_mask_precomp = None
    nn_idx_precomp    = None
    if fill_nn:
        from scipy.ndimage import label as _label
        from scipy.spatial import cKDTree as _cKDTree
        _ri, _ci = np.indices(full_shape)
        _valid_pos = np.column_stack([_ri[~nan_mask_full], _ci[~nan_mask_full]])
        _tree = _cKDTree(_valid_pos)
        _labeled, _n_comp = _label(nan_mask_full)
        _sizes = np.bincount(_labeled.ravel())
        fill_mask_precomp = np.zeros_like(nan_mask_full)
        for _i in range(1, _n_comp + 1):
            if _sizes[_i] <= 1000:
                fill_mask_precomp |= _labeled == _i
        _fill_pos = np.column_stack([_ri[fill_mask_precomp], _ci[fill_mask_precomp]])
        _, nn_idx_precomp = _tree.query(_fill_pos)
        del _ri, _ci, _labeled, _tree  # free memory

    # ---- Sensitivity analysis inner loop ---------------------------------------
    def _run_sensitivity(with_n2000):
        routes    = []
        rs_shape  = None

        for i in range(N):
            if (i + 1) % 20 == 0:
                print(f"    iteration {i + 1}/{N}", flush=True)

            c1 = _add_noise(sw_base, SIGMA)
            c2 = _add_noise(var_base_thresh, SIGMA)
            c3 = _add_noise(amp_base_thresh, SIGMA)
            cb = _add_noise(np.ones(full_shape), SIGMA)

            combined = cm_mod.nansum([cb, c1, c2, c3], axis=0)
            combined[nan_mask_full] = np.nan

            if fill_nn:
                # Fill small NaN holes via precomputed NN indices
                valid_vals = combined[~nan_mask_full]
                combined[fill_mask_precomp] = valid_vals[nn_idx_precomp]
                # Fill any remaining large-hole NaN with max cost
                remaining = np.isnan(combined)
                if remaining.any():
                    combined[remaining] = float(np.nanmax(combined))
            else:
                finite = combined[~np.isnan(combined)]
                combined[np.isnan(combined)] = float(np.nanmax(finite)) if finite.size else 1.0

            if with_n2000:
                combined[n2k_mask] = -1.0

            if RESCALE > 1:
                combined = ski.measure.block_reduce(
                    combined, block_size=RESCALE, func=np.nanmax)
                rs_shape = combined.shape

            planner = AStarPlanner(
                cost_grid=combined, max_turn_steps=1,
                heuristic_weight=1.0, momentum=MOMENTUM,
            )
            try:
                result = planner.solve(
                    start=(sy_rs, sx_rs), goal=(ey_rs, ex_rs),
                    start_heading=None, goal_heading=None,
                )
            except Exception:
                result = None
            if result is not None:
                routes.append(result)

        shape_used = rs_shape if rs_shape is not None else full_shape
        heatmap    = np.zeros(shape_used)
        for route in routes:
            heatmap += route.get_numpy_path()

        # Consensus: A* on inverse-frequency cost
        heatmap_f      = np.where(heatmap == 0, 0.0, heatmap)
        consensus_cost = (N + 1) / (heatmap_f + 1)
        consensus_cost[nan_mask_rs] = float(np.nanmax(consensus_cost))
        if with_n2000:
            consensus_cost[n2k_mask_rs] = -1.0

        try:
            c_planner = AStarPlanner(
                cost_grid=consensus_cost, max_turn_steps=1,
                heuristic_weight=1.0, momentum=MOMENTUM,
            )
            consensus_result = c_planner.solve(
                start=(sy_rs, sx_rs), goal=(ey_rs, ex_rs),
                start_heading=None, goal_heading=None,
            )
        except Exception as exc:
            print(f"    [warn] consensus A* failed: {exc}", flush=True)
            consensus_result = None

        heatmap[heatmap == 0] = np.nan   # non-visited → transparent
        n_ok = len(routes)
        if consensus_result:
            print(f"    {n_ok}/{N} runs succeeded; consensus cost "
                  f"{consensus_result.total_cost:.1f}", flush=True)
        else:
            print(f"    {n_ok}/{N} runs succeeded; consensus route not found",
                  flush=True)
        return heatmap, consensus_result

    print("  Running sensitivity analysis (free) …", flush=True)
    heatmap_free, consensus_free = _run_sensitivity(with_n2000=False)
    print("  Running sensitivity analysis (N2000) …", flush=True)
    heatmap_n2k,  consensus_n2k  = _run_sensitivity(with_n2000=True)

    # ---- Compute shared heatmap vmax for better colormap gradient ---------------
    _all_visits = np.concatenate([
        heatmap_free[~np.isnan(heatmap_free)].ravel(),
        heatmap_n2k[~np.isnan(heatmap_n2k)].ravel(),
    ])
    heat_vmax = max(1, int(np.percentile(_all_visits, 90))) if _all_visits.size > 0 else N

    # ---- Optional NN fill of display bathymetry --------------------------------
    if fill_nn:
        print("  Filling bathymetry NaNs (nearest-neighbour, max_gap=1000) …",
              flush=True)
        bath.fill_nans_nn(max_gap=1000)

    # ---- Downscale all display layers to match heatmap resolution (÷RESCALE) ---
    # Everything is rendered at the rescaled grid — no upscaling artefacts.
    bl     = bath.bl
    extent = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]

    bath_ds  = ski.measure.block_reduce(bath.costs,    block_size=RESCALE, func=np.nanmean)
    sw_prob  = np.clip(
        ski.measure.block_reduce(
            np.where(np.isnan(sw_cm.costs), 0.0, sw_cm.costs),
            block_size=RESCALE, func=np.max),
        0.0, 1.0)
    var_flag = ski.measure.block_reduce(
        ((~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH)).astype(float),
        block_size=RESCALE, func=np.max).astype(bool)
    amp_flag = ski.measure.block_reduce(
        ((~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH)).astype(float),
        block_size=RESCALE, func=np.max).astype(bool)
    # n2k_mask_rs is already at rescaled resolution

    shape = bath_ds.shape   # (rows//RESCALE, cols//RESCALE)

    def _rgba(r, g, b, alpha_arr):
        out = np.zeros((*shape, 4), dtype=float)
        out[..., 0] = r
        out[..., 1] = g
        out[..., 2] = b
        out[..., 3] = alpha_arr
        return out

    sw_rgba  = _rgba(0.84, 0.15, 0.16, sw_prob * SW_ALPHA)
    var_rgba = _rgba(0.80, 0.10, 0.80, np.where(var_flag,    VAR_ALPHA, 0.0))
    amp_rgba = _rgba(1.00, 0.75, 0.00, np.where(amp_flag,    AMP_ALPHA, 0.0))
    n2k_rgba = _rgba(0.50, 0.50, 0.50, np.where(n2k_mask_rs, N2K_ALPHA, 0.0))

    # ---- Rescaled-grid cell → UTM ----------------------------------------------
    cell_m = 100 * RESCALE   # metres per rescaled cell

    def _path_utm(result):
        if result is None:
            return None, None
        xs = [bl[0] + col * cell_m + cell_m / 2 for _, col in result.coords]
        ys = [bl[1] + row * cell_m + cell_m / 2 for row, _ in result.coords]
        return xs, ys

    # ---- Figure (2 rows × 1 col, identical layout to Fig 27) ------------------
    cm_bath2 = copy.copy(CMAP_BATH)
    cm_bath2.set_bad(NAN_COLOR)

    fig, axes = plt.subplots(2, 1, figsize=(16, 13), constrained_layout=True)

    configs = [
        ("free", heatmap_free, consensus_free,
         "Route frequency heatmap — unrestricted", False),
        ("n2k",  heatmap_n2k,  consensus_n2k,
         "Route frequency heatmap — N2000 zones excluded", True),
    ]
    im_bath = None
    im_heat = None

    for ax, (key, heatmap, cons_result, title, show_n2k) in zip(axes, configs):
        # Bathymetry base (already downscaled to heatmap resolution)
        bath_data = bath_ds if fill_nn else np.ma.masked_invalid(bath_ds)
        im = ax.imshow(bath_data, cmap=cm_bath2, origin="lower", extent=extent,
                       interpolation="nearest", aspect="equal", zorder=1)
        if im_bath is None:
            im_bath = im

        # Risk overlays
        for rgba in (sw_rgba, var_rgba, amp_rgba):
            ax.imshow(rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=2)
        if show_n2k and n2k_mask_rs.any():
            ax.imshow(n2k_rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=3)

        # Route-frequency heatmap (native rescaled resolution, no upscaling)
        im_h = ax.imshow(
            heatmap, cmap="YlOrRd", vmin=0, vmax=heat_vmax, alpha=HEAT_ALPHA,
            origin="lower", extent=extent, interpolation="nearest", zorder=4,
        )
        if im_heat is None:
            im_heat = im_h

        # Reference routes
        try:
            import geopandas as gpd
            for fpath, color, rlabel in zip(ROUTE_FILES, ROUTE_COLORS, ROUTE_LABELS):
                if os.path.exists(fpath):
                    gpd.read_file(fpath).to_crs(bath.csr).plot(
                        ax=ax, color=color, linewidth=2.0, zorder=5)
        except ImportError:
            pass

        # Consensus route
        xs_c, ys_c = _path_utm(cons_result)
        if xs_c is not None:
            ax.plot(xs_c, ys_c, color=PATH_COLOR, lw=2.2,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[withStroke(linewidth=4.5, foreground=PATH_EDGE)],
                    zorder=7)

        # Legend
        sw_p   = Patch(facecolor=(0.84, 0.15, 0.16, SW_ALPHA), edgecolor="none",
                       label="Sandwave detection  (prob. weighted)")
        var_p  = Patch(facecolor=(0.80, 0.10, 0.80, VAR_ALPHA), edgecolor="none",
                       label=f"High variance  (log-std > {VAR_THRESH})")
        amp_p  = Patch(facecolor=(1.00, 0.75, 0.00, AMP_ALPHA), edgecolor="none",
                       label=f"High amplitude  (> {AMP_THRESH} m)")
        n2k_p  = Patch(facecolor=(0.50, 0.50, 0.50, N2K_ALPHA), edgecolor="none",
                       label="N2000 exclusion zone  (impassable)")
        heat_p = Patch(facecolor=plt.cm.YlOrRd(0.65), alpha=HEAT_ALPHA,
                       edgecolor="none", label=f"Route frequency  (n = {N})")
        cons_lbl = "Consensus route" if cons_result else "Consensus route (not found)"
        cons_h = Line2D([0], [0], color=PATH_COLOR, lw=2.2,
                        path_effects=[withStroke(linewidth=4.5, foreground=PATH_EDGE)],
                        label=cons_lbl)
        route_hdl = [Line2D([0], [0], color=c, lw=2, label=l)
                     for c, l in zip(ROUTE_COLORS, ROUTE_LABELS)]

        leg_handles = [cons_h, *route_hdl, heat_p, sw_p, var_p, amp_p]
        if not fill_nn:
            leg_handles.append(
                Patch(facecolor=NAN_COLOR, edgecolor="none", label="No survey data"))
        if show_n2k:
            leg_handles.insert(1, n2k_p)
        ax.legend(handles=leg_handles, loc="upper left",
                  fontsize=FS - 4, frameon=True, edgecolor="#cccccc")

        ax.set_title(title, fontsize=FS_TITLE, pad=7, fontweight="bold")
        ax.set_xlabel("Easting  [m, UTM 31N]", fontsize=FS - 2)
        ax.set_ylabel("Northing  [m, UTM 31N]", fontsize=FS - 2)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.locator_params(axis="x", nbins=6)
        ax.locator_params(axis="y", nbins=6)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FS - 3)
        plt.setp(ax.get_yticklabels(), fontsize=FS - 3)

    # Shared colorbars: depth (inner) + route frequency (outer)
    if im_bath is not None:
        cbar = fig.colorbar(im_bath, ax=axes.tolist(), shrink=0.75, pad=0.02)
        cbar.set_label("Depth  [m]", fontsize=FS - 2)
        cbar.ax.tick_params(labelsize=FS - 3)
    if im_heat is not None:
        cbar2 = fig.colorbar(im_heat, ax=axes.tolist(), shrink=0.75, pad=0.10)
        cbar2.set_label(f"Route visits  (out of {N})", fontsize=FS - 2)
        cbar2.ax.tick_params(labelsize=FS - 3)

    fig.suptitle(
        f"A* sensitivity analysis — route frequency  "
        f"(n = {N},  \u03c3 = {SIGMA},  momentum = {MOMENTUM},  "
        f"rescale \xd7{RESCALE},  100 m cells)",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    _save_or_show(fig, save_path)


def plot_sensitivity_route_nn(save_path=None):
    """Like plot_sensitivity_route but with NN-filled cost map and bathymetry."""
    plot_sensitivity_route(save_path=save_path, fill_nn=True)


# ---------------------------------------------------------------------------
# Cover art — A4 portrait, both scenarios, full-res + bilinear rendering
# ---------------------------------------------------------------------------

def plot_cover_art(save_path=None):
    """
    Thesis cover art based on Fig 28b — A4 portrait with both scenarios stacked.

    Quality improvements over the regular figure:
      - Background layers at full 100 m resolution (350x800) rather than the
        4x rescaled 400 m grid, giving 4x more spatial detail.
      - Heatmap upsampled from the rescaled grid to full resolution with bilinear
        zoom so there are no visible square pixels.
      - All imshow calls use interpolation="bilinear" for smooth rendering.
      - Heatmap alpha is proportional to sqrt(frequency) so unvisited cells fade
        out gracefully.
    """
    import skimage as ski
    import costmap as cm_mod
    from Astar import AStarPlanner
    from matplotlib.patheffects import withStroke

    N          = 100
    SIGMA      = 0.5
    VAR_THRESH = 0.05
    AMP_THRESH = 0.08
    RESCALE    = 4
    MOMENTUM   = 2

    LABELS_DIR = "sandwave_detection_v8/labels"
    VAR_DIR    = "variance_rasters/Rasters"
    AMP_DIR    = "amplitude_rasters/Rasters_amp"
    N2K_SHP    = "shapes/n200.shp"

    SW_ALPHA   = 0.45
    VAR_ALPHA  = 0.40
    AMP_ALPHA  = 0.45
    N2K_ALPHA  = 0.60
    HEAT_ALPHA = 0.70

    ROUTE_FILES  = ["shapes/line1.shp", "shapes/line2.shp"]
    ROUTE_COLORS = ["#aeea00", "#ff7f0e"]
    PATH_COLOR   = "white"
    PATH_EDGE    = "#111111"

    def _load_dir(directory, suffix=None):
        d = {}
        for fname in os.listdir(directory):
            if suffix and not fname.endswith(suffix):
                continue
            cid = fname.split("_")[1]
            arr = np.load(os.path.join(directory, fname))[::-1, :]
            d.setdefault(cid, []).append(arr)
        return d

    def _combine(rasters, func=np.nanmean):
        stacked = np.stack(rasters, axis=0)
        with np.errstate(invalid="ignore"):
            return func(stacked, axis=0)

    def _add_noise(arr, sigma):
        noisy = arr + np.random.normal(0, sigma, arr.shape)
        return np.where((~np.isnan(arr)) & (arr != 0), noisy, arr)

    # ---- Load data layers -------------------------------------------------------
    print("  Building bathymetry …", flush=True)
    bath = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir("destriped_rasters").items():
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        xs, xe, ys, ye = bath.slice_cost_map(int(cid))
        bath.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    print("  Loading sandwave labels …", flush=True)
    sw_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(LABELS_DIR,
                                   suffix="destriped_labels_smoothed.npy").items():
        rasters = [r.astype(float) for r in rasters]
        for r in rasters:
            r[r == -1] = np.nan
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.max)
        xs, xe, ys, ye = sw_cm.slice_cost_map(int(cid))
        sw_cm.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    print("  Loading variance …", flush=True)
    var_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(VAR_DIR).items():
        avg = _combine(rasters, np.nanmean)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmean)
        ts  = np.log10(np.sqrt(np.maximum(rs, 0.0)) + 1.0)
        xs, xe, ys, ye = var_cm.slice_cost_map(int(cid))
        var_cm.add_cost(xs, ys, cost=ts, x_idx_end=xe, y_idx_end=ye)

    print("  Loading amplitude …", flush=True)
    amp_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=np.nan)
    for cid, rasters in _load_dir(AMP_DIR).items():
        avg = _combine(rasters, np.nanmax)
        rs  = ski.measure.block_reduce(avg, block_size=5, func=np.nanmax)
        xs, xe, ys, ye = amp_cm.slice_cost_map(int(cid))
        amp_cm.add_cost(xs, ys, cost=rs, x_idx_end=xe, y_idx_end=ye)

    full_shape    = bath.costs.shape
    nan_mask_full = np.isnan(bath.costs)

    sw_base         = sw_cm.costs.copy()
    var_base_thresh = np.where(
        (~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH), 1.0, np.nan)
    amp_base_thresh = np.where(
        (~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH), 1.0, np.nan)

    # N2000 mask
    n2k_mask = np.zeros(full_shape, dtype=bool)
    try:
        arr = np.zeros(full_shape, dtype=float)
        cm_mod.CostMap(dx=100, dy=100, default_cost=0.0).block_n2000(
            array=arr, path=N2K_SHP)
        n2k_mask = (arr == -1)
    except Exception as exc:
        print(f"  [warn] N2000 mask: {exc}", flush=True)

    # Start / goal
    ref_cm = cm_mod.CostMap(dx=100, dy=100, default_cost=1)
    sx, sy = ref_cm.start_utm
    ex, ey = ref_cm.end_utm
    sx_idx, sy_idx = ref_cm.get_idx_from_coordinates(sx, sy)
    ex_idx, ey_idx = ref_cm.get_idx_from_coordinates(ex, ey)
    sx_rs, sy_rs = sx_idx // RESCALE, sy_idx // RESCALE
    ex_rs, ey_rs = ex_idx // RESCALE, ey_idx // RESCALE

    nan_mask_rs = ski.measure.block_reduce(
        nan_mask_full.astype(float), block_size=RESCALE, func=np.max).astype(bool)
    n2k_mask_rs = ski.measure.block_reduce(
        n2k_mask.astype(float), block_size=RESCALE, func=np.max).astype(bool)

    # Precompute NN fill for SMALL NaN holes only (max 1000 px).
    # Large no-data regions stay impassable (filled with max cost, not NN).
    # This matches the original plot_sensitivity_route fill_nn=True logic.
    from scipy.ndimage import zoom as _zoom
    from scipy.ndimage import label as _label
    from scipy.spatial import cKDTree as _cKDTree

    _ri, _ci = np.indices(full_shape)
    _valid_pos = np.column_stack([_ri[~nan_mask_full], _ci[~nan_mask_full]])
    _tree = _cKDTree(_valid_pos)
    _labeled, _n_comp = _label(nan_mask_full)
    _sizes = np.bincount(_labeled.ravel())
    _fill_mask = np.zeros_like(nan_mask_full, dtype=bool)
    for _i in range(1, _n_comp + 1):
        if _sizes[_i] <= 1000:
            _fill_mask |= (_labeled == _i)
    _fill_pos = np.column_stack([_ri[_fill_mask], _ci[_fill_mask]])
    _, _nn_idx_small = _tree.query(_fill_pos)
    del _ri, _ci, _labeled, _tree   # free memory

    # ---- Sensitivity analysis (both scenarios) ---------------------------------
    def _run_sensitivity(with_n2000):
        routes = []
        for i in range(N):
            if (i + 1) % 20 == 0:
                print(f"    iteration {i + 1}/{N}", flush=True)
            c1 = _add_noise(sw_base, SIGMA)
            c2 = _add_noise(var_base_thresh, SIGMA)
            c3 = _add_noise(amp_base_thresh, SIGMA)
            cb = _add_noise(np.ones(full_shape), SIGMA)

            combined = cm_mod.nansum([cb, c1, c2, c3], axis=0)
            combined[nan_mask_full] = np.nan
            # Small holes: NN-fill from nearest valid neighbour
            valid_vals = combined[~nan_mask_full]
            combined[_fill_mask] = valid_vals[_nn_idx_small]
            # Large holes: high cost (impassable after rescale → NaN → A* skips)
            remaining = np.isnan(combined)
            if remaining.any():
                combined[remaining] = float(np.nanmax(combined[~remaining]))
            if with_n2000:
                combined[n2k_mask] = -1.0
            if RESCALE > 1:
                combined = ski.measure.block_reduce(
                    combined, block_size=RESCALE, func=np.nanmax)

            planner = AStarPlanner(
                cost_grid=combined, max_turn_steps=1,
                heuristic_weight=1.0, momentum=MOMENTUM,
            )
            try:
                result = planner.solve(
                    start=(sy_rs, sx_rs), goal=(ey_rs, ex_rs),
                    start_heading=None, goal_heading=None,
                )
            except Exception:
                result = None
            if result is not None:
                routes.append(result)

        rs_shape = routes[0].get_numpy_path().shape if routes else nan_mask_rs.shape
        heatmap  = np.zeros(rs_shape)
        for route in routes:
            heatmap += route.get_numpy_path()

        heatmap_f      = np.where(heatmap == 0, 0.0, heatmap)
        consensus_cost = (N + 1) / (heatmap_f + 1)
        consensus_cost[nan_mask_rs] = float(np.nanmax(consensus_cost))
        if with_n2000:
            consensus_cost[n2k_mask_rs] = -1.0

        try:
            c_planner = AStarPlanner(
                cost_grid=consensus_cost, max_turn_steps=1,
                heuristic_weight=1.0, momentum=MOMENTUM,
            )
            consensus_result = c_planner.solve(
                start=(sy_rs, sx_rs), goal=(ey_rs, ex_rs),
                start_heading=None, goal_heading=None,
            )
        except Exception as exc:
            print(f"    [warn] consensus A* failed: {exc}", flush=True)
            consensus_result = None

        heatmap[heatmap == 0] = np.nan
        print(f"    {len(routes)}/{N} runs succeeded", flush=True)
        return heatmap, consensus_result

    print("  Running sensitivity analysis (free) …", flush=True)
    heatmap_free, consensus_free = _run_sensitivity(False)
    print("  Running sensitivity analysis (N2000) …", flush=True)
    heatmap_n2k,  consensus_n2k  = _run_sensitivity(True)

    # ---- NN-fill bathymetry; use full 100 m resolution for display -------------
    bath.fill_nans_nn(max_gap=1000)
    bath_full = bath.costs   # (350, 800)

    bl     = bath.bl
    extent = [bl[0], bl[0] + 80_000, bl[1], bl[1] + 35_000]
    cell_m = 100 * RESCALE

    # Full-resolution overlays (no block_reduce) — 4× sharper than before
    sw_prob_full  = np.clip(np.where(np.isnan(sw_cm.costs), 0.0, sw_cm.costs), 0.0, 1.0)
    var_flag_full = (~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH)
    amp_flag_full = (~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH)

    def _rgba_full(r, g, b, alpha_arr):
        out = np.zeros((*full_shape, 4), dtype=float)
        out[..., 0] = r; out[..., 1] = g; out[..., 2] = b; out[..., 3] = alpha_arr
        return out

    sw_rgba  = _rgba_full(0.84, 0.15, 0.16, sw_prob_full * SW_ALPHA)
    var_rgba = _rgba_full(0.80, 0.10, 0.80, np.where(var_flag_full, VAR_ALPHA, 0.0))
    amp_rgba = _rgba_full(1.00, 0.75, 0.00, np.where(amp_flag_full, AMP_ALPHA, 0.0))
    n2k_rgba = _rgba_full(0.50, 0.50, 0.50, np.where(n2k_mask,      N2K_ALPHA, 0.0))

    # Upsample heatmaps from RESCALE grid to full res; alpha fades with frequency
    def _heatmap_rgba(hm):
        hm_nonan = np.where(np.isnan(hm), 0.0, hm)
        zr = full_shape[0] / hm.shape[0]
        zc = full_shape[1] / hm.shape[1]
        hm_up    = _zoom(hm_nonan, (zr, zc), order=1)
        _visits  = hm_nonan[hm_nonan > 0]
        vmax     = max(1, int(np.percentile(_visits, 90))) if _visits.size else 1
        norm_hm  = np.clip(hm_up / vmax, 0.0, 1.0)
        rgba     = plt.cm.YlOrRd(norm_hm)
        rgba[..., 3] = HEAT_ALPHA * norm_hm ** 0.5   # smooth alpha falloff
        return rgba

    hm_free_rgba = _heatmap_rgba(heatmap_free)
    hm_n2k_rgba  = _heatmap_rgba(heatmap_n2k)

    def _path_utm(result):
        if result is None:
            return None, None
        xs = [bl[0] + col * cell_m + cell_m / 2 for _, col in result.coords]
        ys = [bl[1] + row * cell_m + cell_m / 2 for row, _ in result.coords]
        return xs, ys

    # ---- Figure — A4 portrait, 2 panels, no decorations -----------------------
    # NaN → white so survey gaps blend seamlessly with the white page background
    cm_bath2 = copy.copy(CMAP_BATH)
    cm_bath2.set_bad("white")

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor="white")

    # Maps bleed to the left and right edges (no side margins).
    # Equal top/bottom margins centre the two panels vertically on the page.
    # GAP = 0 so the two maps touch directly with no white stripe between them.
    ML, MR = 0.0,  0.0
    MT, MB = 0.08, 0.08
    panel_w = 1.0 - ML - MR
    panel_h = (1.0 - MT - MB) / 2.0

    ax_free = fig.add_axes([ML, MB + panel_h, panel_w, panel_h])
    ax_n2k  = fig.add_axes([ML, MB,           panel_w, panel_h])

    configs = [
        (ax_free, hm_free_rgba, consensus_free, False),
        (ax_n2k,  hm_n2k_rgba,  consensus_n2k,  True),
    ]

    for ax, hm_rgba, cons_result, show_n2k in configs:
        ax.imshow(bath_full, cmap=cm_bath2, origin="lower", extent=extent,
                  interpolation="bilinear", aspect="auto", zorder=1)
        for rgba in (sw_rgba, var_rgba, amp_rgba):
            ax.imshow(rgba, origin="lower", extent=extent,
                      interpolation="bilinear", zorder=2)
        if show_n2k:
            ax.imshow(n2k_rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=3)
        ax.imshow(hm_rgba, origin="lower", extent=extent,
                  interpolation="bilinear", zorder=4)

        try:
            import geopandas as gpd
            for fpath, color in zip(ROUTE_FILES, ROUTE_COLORS):
                if os.path.exists(fpath):
                    gpd.read_file(fpath).to_crs(bath.csr).plot(
                        ax=ax, color=color, linewidth=2.0, zorder=5)
        except ImportError:
            pass

        xs_c, ys_c = _path_utm(cons_result)
        if xs_c is not None:
            ax.plot(xs_c, ys_c, color=PATH_COLOR, lw=2.2,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[withStroke(linewidth=4.5, foreground=PATH_EDGE)],
                    zorder=7)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.margins(0)
        ax.set_axis_off()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 11b — Angle-detection failure: cell 93 / CDI 3466117
# ---------------------------------------------------------------------------

_ANGLE_FAIL_PATH = "rasters/cell_93_CDI_3466117.npy"
_ANGLE_FAIL_LABEL = "cell_93  ·  CDI 3466117\n(automatic angle detection failed — manual override required)"


def plot_angle_detection_failure(save_path=None):
    """
    Five-panel diagnostic showing why automatic stripe-angle detection fails
    for cell 93 / CDI 3466117.

    Layout (2 rows):
      Row 0 — three image panels:
        Col 0 : Bathymetric survey data (NaN regions shown in grey).
        Col 1 : Detrended residuals (Gaussian sigma = DETREND_SIGMA_CHOSEN).
        Col 2 : Windowed + zero-padded FFT log-amplitude.
      Row 1 — two line-plot panels:
        Col 0+1 : Absolute notch-band response vs. rotation angle (0-179°).
        Col 2+3 : Relative response (absolute minus local-neighbour mean, ±8°).
                  The automatically selected peak is marked with a dashed line.
    """
    print("  Loading raster…", flush=True)
    raw = np.load(_ANGLE_FAIL_PATH)
    filled, mean_val, nan_mask = _fill_raster_path(_ANGLE_FAIL_PATH)

    _, residuals = _gaussian_detrend(filled, DETREND_SIGMA_CHOSEN)

    h, w     = residuals.shape
    window   = np.outer(np.hanning(h), np.hanning(w))
    eps      = 1e-2
    win_clip = np.where(window < eps, eps, window)
    pad      = max(h, w) // 2
    win_padded = np.pad(residuals * win_clip, pad, mode="constant")
    F_win      = np.fft.fftshift(np.fft.fft2(win_padded))
    log_amp    = np.log(np.abs(F_win) + 1)

    print("  Angle sweep…", flush=True)
    auto_angle, resp, rel = _find_stripe_angle(log_amp)
    print(f"  Auto-detected angle: {auto_angle:.0f}°", flush=True)
    angles = np.arange(0, 180, 1)

    # Build the detected-angle notch overlay (same as plot_angle_detection)
    hp, wp = win_padded.shape
    base   = np.zeros((hp, wp), dtype=float)
    for i in range(-NOTCH_WIDTH_CHOSEN, NOTCH_WIDTH_CHOSEN + 1):
        base += np.eye(hp, wp, k=i)
    cy, cx = hp // 2, wp // 2
    base[cy - NOTCH_CENTER_SIZE:cy + NOTCH_CENTER_SIZE,
         cx - NOTCH_CENTER_SIZE:cx + NOTCH_CENTER_SIZE] = 0
    base  = np.clip(base, 0, 1)
    notch_rot = ndimage.rotate(base, float(auto_angle), reshape=False)
    notch_rot = np.clip(notch_rot, 0, 1)

    # ---- Layout: 2 rows, 4 logical columns (images share 1 unit, plots 1.5)
    fig = plt.figure(figsize=(22, 11))
    gs  = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1.5, 1.5],
        hspace=0.28, wspace=0.28,
        left=0.07, right=0.98, top=0.90, bottom=0.09,
    )

    ax_data  = fig.add_subplot(gs[0, 0])
    ax_res   = fig.add_subplot(gs[0, 1])
    ax_fft   = fig.add_subplot(gs[0, 2:])     # FFT spans the two right columns
    ax_abs   = fig.add_subplot(gs[1, 0:2])    # absolute response spans left cols
    ax_rel   = fig.add_subplot(gs[1, 2:])     # relative response spans right cols

    # Show raw data with NaN regions in grey
    raw_disp = raw.copy()
    cmap_data = plt.get_cmap("cmo.deep").copy()
    cmap_data.set_bad(color="lightgrey")
    vmin_d, vmax_d = float(np.nanpercentile(raw, 2)), float(np.nanpercentile(raw, 98))
    im0 = ax_data.imshow(raw_disp, cmap=cmap_data, origin="upper",
                         vmin=vmin_d, vmax=vmax_d)
    cb0 = fig.colorbar(im0, ax=ax_data, fraction=0.046, pad=0.02)
    cb0.set_label("Depth  [m]", fontsize=FS - 2)
    cb0.ax.tick_params(labelsize=FS - 4)
    ax_data.set_title("Bathymetric data\n(grey = no data)", fontsize=FS)

    # Residuals panel
    res_lim = float(np.nanpercentile(np.abs(residuals), 98))
    im1 = ax_res.imshow(residuals + mean_val, cmap="cmo.deep", origin="upper",
                        vmin=-res_lim + mean_val, vmax=res_lim + mean_val)
    cb1 = fig.colorbar(im1, ax=ax_res, fraction=0.046, pad=0.02)
    cb1.set_label("Residual  [m]", fontsize=FS - 2)
    cb1.ax.tick_params(labelsize=FS - 4)
    ax_res.set_title(f"Detrended residuals\n(Gaussian σ = {DETREND_SIGMA_CHOSEN} px)", fontsize=FS)

    # FFT log-amplitude + notch overlay
    fft_max = float(log_amp.max())
    im2 = ax_fft.imshow(log_amp, cmap="hot", origin="upper", vmin=0, vmax=fft_max)
    # Overlay notch in orange
    notch_rgba              = np.zeros((*notch_rot.shape, 4), dtype=float)
    notch_rgba[..., 0]      = 1.0          # R
    notch_rgba[..., 1]      = 0.55         # G  -> orange
    notch_rgba[..., 3]      = np.clip(notch_rot * 0.6, 0, 1)
    ax_fft.imshow(notch_rgba, origin="upper")
    cb2 = fig.colorbar(im2, ax=ax_fft, fraction=0.03, pad=0.02)
    cb2.set_label("log |F|", fontsize=FS - 2)
    cb2.ax.tick_params(labelsize=FS - 4)
    ax_fft.set_title(
        f"FFT log amplitude (windowed + zero-padded)\n"
        f"orange band = notch at auto-detected angle {auto_angle:.0f}°",
        fontsize=FS,
    )
    ax_fft.set_xlabel("Frequency (px)", fontsize=FS - 2)

    # Absolute response vs angle
    ax_abs.plot(angles, resp, color="steelblue", lw=1.5)
    ax_abs.axvline(auto_angle, color=HIGHLIGHT_COLOR, lw=1.8, ls="--",
                   label=f"Auto-detected: {auto_angle:.0f}°")
    ax_abs.set_xlabel("Rotation angle  [°]", fontsize=FS - 1)
    ax_abs.set_ylabel("Mean log |F| inside notch", fontsize=FS - 1)
    ax_abs.set_title("Absolute notch-band response", fontsize=FS)
    ax_abs.set_xlim(0, 179)
    ax_abs.legend(fontsize=FS - 2)
    ax_abs.grid(alpha=0.3)
    ax_abs.tick_params(labelsize=FS - 3)

    # Relative response vs angle
    ax_rel.plot(angles, rel, color="steelblue", lw=1.5)
    ax_rel.axvline(auto_angle, color=HIGHLIGHT_COLOR, lw=1.8, ls="--",
                   label=f"Auto-detected: {auto_angle:.0f}°")
    ax_rel.axhline(0, color="grey", lw=0.8, ls=":")
    ax_rel.set_xlabel("Rotation angle  [°]", fontsize=FS - 1)
    ax_rel.set_ylabel("Relative response  [a.u.]", fontsize=FS - 1)
    ax_rel.set_title("Relative response (absolute − local-neighbour mean,  ±8°)", fontsize=FS)
    ax_rel.set_xlim(0, 179)
    ax_rel.legend(fontsize=FS - 2)
    ax_rel.grid(alpha=0.3)
    ax_rel.tick_params(labelsize=FS - 3)

    fig.suptitle(_ANGLE_FAIL_LABEL, fontsize=FS + 1, fontweight="bold", y=0.97)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved -> {save_path}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure — Annual seabed change rate along the Route 1 corridor
# ---------------------------------------------------------------------------

def plot_route1_amplitude_along_line(save_path=None, n_per_row=6):
    """
    Strip-chart of mean signed annual seabed change rate (m/yr) along Route 1.

    For each 5x5 km cell that Route 1 passes through:
      - All signed difference rasters in difference_rasters/Rasters_diff for that
        cell are averaged (mean across survey pairs) to produce one change-rate image.
      - Cells are ordered by chainage (km along route geometry).
      - The Route 1 pipeline is overlaid on each panel in pixel coordinates.
      - The strip is wrapped into rows of n_per_row cells.

    Diverging colormap: blue = accretion, red = erosion, white = stable.
    """
    import geopandas as gpd
    from shapely.geometry import box, Point
    from shapely.ops import linemerge, unary_union

    DIFF_DIR  = "difference_rasters/Rasters_diff"
    LINE_PATH = "shapes/line1.shp"
    BL        = (555652, 5910512)   # bottom-left corner of cost-map grid (UTM 31N)
    CELL_KM   = 5000                # 5 km cells
    N_COLS    = 16
    N_ROWS    = 7
    LINE_COLOR = "#ffffff"          # white pipeline line

    # ---------------------------------------------------------------- route --
    line_gdf   = gpd.read_file(LINE_PATH).to_crs("EPSG:32631")
    route_geom = unary_union(line_gdf.geometry.values)
    if route_geom.geom_type == "MultiLineString":
        merged     = linemerge(route_geom)
        route_geom = merged if merged.geom_type == "LineString" else route_geom

    # ------------------------------------------------------ find cells -------
    intersecting = []
    for cell_id in range(N_COLS * N_ROWS):
        row = cell_id // N_COLS
        col = cell_id % N_COLS
        x0  = BL[0] + col * CELL_KM
        y0  = BL[1] + row * CELL_KM
        cell_poly = box(x0, y0, x0 + CELL_KM, y0 + CELL_KM)
        if route_geom.intersects(cell_poly):
            d = route_geom.project(Point(x0 + CELL_KM / 2, y0 + CELL_KM / 2))
            intersecting.append((d, cell_id))

    intersecting.sort()
    cell_order    = [cid       for _, cid in intersecting]
    cell_dists_km = [d / 1000  for d, _   in intersecting]

    # ------------------------------------------- group diff raster files -----
    diff_files: dict = {}
    for fname in os.listdir(DIFF_DIR):
        if not fname.endswith(".npy"):
            continue
        try:
            cid = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue
        diff_files.setdefault(cid, []).append(os.path.join(DIFF_DIR, fname))

    # --------------------------------- load + average per cell ---------------
    cell_rasters: dict = {}
    for cid in cell_order:
        if cid not in diff_files:
            cell_rasters[cid] = None
            continue
        arrays  = [np.load(f) for f in diff_files[cid]]
        stacked = np.stack(arrays, axis=0)
        cell_rasters[cid] = np.nanmean(stacked, axis=0)

    # ----------------------------------------- symmetric colour limits -------
    all_vals = np.concatenate([
        r.ravel() for r in cell_rasters.values() if r is not None
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    vlim = float(np.nanpercentile(np.abs(all_vals), 99)) if len(all_vals) else 1.0

    # ---------------------------------------------------- layout -------------
    n_cells        = len(cell_order)
    n_rows_layout  = int(np.ceil(n_cells / n_per_row))
    panel_w, panel_h = 2.5, 2.5
    fig_w = panel_w * n_per_row + 1.8
    fig_h = panel_h * n_rows_layout + 1.2

    fig, axes = plt.subplots(
        n_rows_layout, n_per_row,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    cmap_div = copy.copy(cmocean.cm.balance)
    cmap_div.set_bad(NAN_COLOR)

    im_ref = None
    for idx, (cid, km) in enumerate(zip(cell_order, cell_dists_km)):
        r  = idx // n_per_row
        c  = idx %  n_per_row
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])

        # Cell UTM bounds
        cell_col = cid % N_COLS
        cell_row = cid // N_COLS
        x_min = BL[0] + cell_col * CELL_KM
        y_min = BL[1] + cell_row * CELL_KM
        x_max = x_min + CELL_KM
        y_max = y_min + CELL_KM

        raster = cell_rasters[cid]
        if raster is None:
            ax.set_facecolor(NAN_COLOR)
            ax.set_title(f"Cell {cid}\n{km:.0f} km", fontsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)
            # Still draw the pipeline on empty cells
            _overlay_pipeline(ax, route_geom, x_min, y_min, x_max, y_max,
                               raster_h=250, raster_w=250,
                               color=LINE_COLOR)
            # Fix axis limits to match the 250×250 pixel grid (no imshow to anchor them)
            ax.set_xlim(-0.5, 249.5)
            ax.set_ylim(249.5, -0.5)   # inverted: row 0 = north = top
            continue

        h, w = raster.shape
        im = ax.imshow(
            np.ma.masked_invalid(raster),
            cmap=cmap_div,
            vmin=-vlim, vmax=vlim,
            origin="upper",
            interpolation="nearest",
        )
        im_ref = im
        ax.set_title(f"Cell {cid}   {km:.0f} km", fontsize=7, pad=3)

        # Pipeline overlay
        _overlay_pipeline(ax, route_geom, x_min, y_min, x_max, y_max,
                          raster_h=h, raster_w=w,
                          color=LINE_COLOR)

    # Row labels (km range)
    for r in range(n_rows_layout):
        start_idx = r * n_per_row
        end_idx   = min(start_idx + n_per_row - 1, n_cells - 1)
        axes[r, 0].set_ylabel(
            f"{cell_dists_km[start_idx]:.0f}–{cell_dists_km[end_idx]:.0f} km",
            fontsize=8, labelpad=4,
        )

    # Hide unused axes in the last row
    used_last_row = n_cells % n_per_row
    if used_last_row:
        for j in range(used_last_row, n_per_row):
            axes[n_rows_layout - 1, j].axis("off")

    # Shared colorbar
    if im_ref is not None:
        fig.subplots_adjust(
            left=0.06, right=0.88, top=0.91, bottom=0.04,
            hspace=0.40, wspace=0.08,
        )
        cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.72])
        cbar    = fig.colorbar(im_ref, cax=cbar_ax, extend="both")
        cbar.set_label("Annual change rate  [m/yr]\n(mean across survey pairs)",
                       fontsize=FS - 2, labelpad=6)
        cbar.ax.tick_params(labelsize=FS - 3)
    else:
        fig.tight_layout()

    # Legend for pipeline line
    from matplotlib.lines import Line2D
    legend_handle = Line2D([0], [0], color=LINE_COLOR, linewidth=2, label="Route 1")
    fig.legend(handles=[legend_handle], loc="lower right",
               bbox_to_anchor=(0.88, 0.04), fontsize=FS - 2, framealpha=0.9)

    fig.suptitle(
        "Route 1 corridor — mean signed annual seabed change rate per 5×5 km cell",
        fontsize=FS_TITLE, fontweight="bold", y=0.97,
    )

    _save_or_show(fig, save_path)


def _overlay_pipeline(ax, route_geom, x_min, y_min, x_max, y_max,
                      raster_h, raster_w, color="#ffffff", lw=3.0):
    """Clip route_geom to the cell box and draw it in pixel coordinates on ax.

    origin='upper': row 0 is the north edge (y_max), so
      py = (y_max - y_utm) / (CELL_KM / raster_h)
      px = (x_utm - x_min) / (CELL_KM / raster_w)
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    cell_box  = box(x_min, y_min, x_max, y_max)
    clipped   = route_geom.intersection(cell_box)
    if clipped.is_empty:
        return

    x_scale = raster_w / (x_max - x_min)   # px per metre
    y_scale = raster_h / (y_max - y_min)   # px per metre

    def _utm_to_px(coords):
        xs = [(x - x_min) * x_scale for x, _ in coords]
        ys = [(y_max - y) * y_scale for _, y in coords]
        return xs, ys

    geoms = (clipped.geoms if hasattr(clipped, "geoms") else [clipped])
    for geom in geoms:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            px, py = _utm_to_px(geom.coords)
            ax.plot(px, py, color=color, linewidth=lw, solid_capstyle="round")
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                px, py = _utm_to_px(part.coords)
                ax.plot(px, py, color=color, linewidth=lw, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Figure — Cell 22 temporal comparison: 2023 bathymetry + 2023 − 2004 diff
# ---------------------------------------------------------------------------

def plot_cell22_temporal_comparison(save_path=None):
    """
    Two-panel figure for cell 22 using raw (undestriped) rasters to motivate
    the destriping preprocessing step:
      Left  : raw bathymetry 2023 (CDI 3844672), cmocean.deep colormap.
              Stripe artefacts from multibeam acquisition are clearly visible.
      Right : depth change 2023 − 2004 (CDI 2174760 baseline), RdBu_r diverging.
              Without destriping, stripe noise from both surveys aliases into
              the difference map and masks real seabed change.
    """
    path_2023 = "rasters/cell_22_CDI_3844672.npy"
    path_2004 = "rasters/cell_22_CDI_2174760.npy"

    r2023 = np.load(path_2023)
    r2004 = np.load(path_2004)

    nan_2023 = np.isnan(r2023)
    nan_2004 = np.isnan(r2004)
    nan_diff = nan_2023 | nan_2004

    diff     = r2023 - r2004
    vmin     = float(np.nanmin(r2023))
    vmax     = float(np.nanmax(r2023))
    sym_diff = float(np.nanmax(np.abs(diff[~nan_diff])))

    cm_bath = copy.copy(CMAP_BATH);      cm_bath.set_bad(NAN_COLOR)
    cm_diff = copy.copy(plt.cm.RdBu_r);  cm_diff.set_bad(NAN_COLOR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.95, top=0.78, bottom=0.05)

    # Left panel — raw 2023 bathymetry
    im0 = axes[0].imshow(
        np.ma.masked_where(nan_2023, r2023),
        cmap=cm_bath, origin="upper", vmin=vmin, vmax=vmax,
    )
    axes[0].set_title("Raw bathymetry  —  2023\n(cell 22,  CDI 3844672)",
                      fontsize=FS_TITLE, pad=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("Depth (m)", fontsize=FS)
    cb0.ax.tick_params(labelsize=FS - 2)

    # Right panel — raw difference map
    im1 = axes[1].imshow(
        np.ma.masked_where(nan_diff, diff),
        cmap=cm_diff, origin="upper", vmin=-sym_diff, vmax=sym_diff,
    )
    axes[1].set_title(r"Raw depth change  2023 $-$ 2004" + "\n(CDI 3844672 − CDI 2174760)",
                      fontsize=FS_TITLE, pad=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label(r"$\Delta$ depth (m)", fontsize=FS)
    cb1.ax.tick_params(labelsize=FS - 2)

    fig.suptitle(
        "Cell 22  —  raw (undestriped) data  (5 km × 5 km,  20 m resolution)",
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
    # plot_angle_detection          (save_path=os.path.join(SAVE_DIR, "11_angle_detection.png"))
    # plot_angle_detection_failure  (save_path=os.path.join(SAVE_DIR, "11b_angle_detection_failure.png"))
    # plot_windowed_vs_unwindowed_filtering(save_path=os.path.join(SAVE_DIR, "12_windowed_vs_unwindowed.png"))
    # plot_swd_residuals        (save_path=os.path.join(SAVE_DIR, "13_swd_residuals.png"))
    # plot_swd_local_std        (save_path=os.path.join(SAVE_DIR, "14_swd_local_std.png"))
    # plot_swd_gradient_features(save_path=os.path.join(SAVE_DIR, "15_swd_gradient_features.png"))
    # plot_swd_closing_effect      (save_path=os.path.join(SAVE_DIR, "20_swd_closing_effect.png"))
    # plot_swd_erosion_effect      (save_path=os.path.join(SAVE_DIR, "21_swd_erosion_effect.png"))
    # plot_swd_data_overview       (save_path=os.path.join(SAVE_DIR, "16_swd_data_overview.png"))
    # plot_swd_label_pipeline      (save_path=os.path.join(SAVE_DIR, "17_swd_label_pipeline.png"))
    # plot_swd_sigma_comparison    (save_path=os.path.join(SAVE_DIR, "18_swd_sigma_comparison.png"))
    # plot_swd_threshold_morphology(save_path=os.path.join(SAVE_DIR, "19_swd_threshold_morphology.png"))
    # plot_coverage_and_sandwaves(save_path=os.path.join(SAVE_DIR, "22_coverage_sandwaves.png"))
    # plot_variance_costmap (save_path=os.path.join(SAVE_DIR, "23_variance_costmap.png"))
    # plot_amplitude_costmap(save_path=os.path.join(SAVE_DIR, "24_amplitude_costmap.png"))
    # plot_local_std_mask_cell56(save_path=os.path.join(SAVE_DIR, "25_local_std_mask_cell56.png"))
    # plot_local_std_mask_cell39(save_path=os.path.join(SAVE_DIR, "25b_local_std_mask_cell39.png"))
    # plot_combined_costmaps(save_path=os.path.join(SAVE_DIR, "26_combined_costmaps.png"))
    # plot_optimised_route(save_path=os.path.join(SAVE_DIR, "27_optimised_route.png"))
    # plot_optimised_route_nn(save_path=os.path.join(SAVE_DIR, "27b_optimised_route_nn.png"))
    # plot_sensitivity_route(save_path=os.path.join(SAVE_DIR, "28_sensitivity_route.png"))
    # plot_sensitivity_route_nn(save_path=os.path.join(SAVE_DIR, "28b_sensitivity_route_nn.png"))
    # plot_angle_detection_failure(save_path=os.path.join(SAVE_DIR, "11b_angle_detection_failure.png"))
    # plot_route1_amplitude_along_line(save_path=os.path.join(SAVE_DIR, "29_route1_amplitude_along_line.png"))
    # plot_cell22_temporal_comparison(save_path=os.path.join(SAVE_DIR, "30_cell22_temporal_comparison.png"))
    plot_cover_art(save_path=os.path.join(SAVE_DIR, "cover_art.png"))
    print(f"\nAll plots saved to: {SAVE_DIR}/")
