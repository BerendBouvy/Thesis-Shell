"""
cover_art.py
------------
Thesis cover art: immersive 3D view of the Dutch North Sea seabed
with sand wave morphology and the A* optimal pipeline route.

Run from the project root:
    .venv/Scripts/python cover_art.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: registers '3d' projection
from matplotlib.colors import LightSource, LinearSegmentedColormap
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
import skimage.measure as ski_measure
import geopandas as gpd

sys.path.insert(0, os.path.dirname(__file__))
import costmap as cm_mod
from Astar import AStarPlanner

# ── Configuration ─────────────────────────────────────────────────────────────
DX, DY       = 100, 100
BL           = (555652, 5910512)      # bottom-left, UTM 31N
LABELS_DIR   = "sandwave_detection_v8/labels"
DESTRIPED    = "destriped_rasters"
VAR_DIR      = "variance_rasters/Rasters"
AMP_DIR      = "amplitude_rasters/Rasters_amp"

VAR_THRESH   = 0.05
AMP_THRESH   = 0.08
MOMENTUM     = 8

VE           = 260    # vertical exaggeration
STRIDE       = 2      # spatial downsampling (2 → 175 × 400 mesh)
SMOOTH_SIGMA = 7      # blur for gap fill (pixels at 100m res = 700m smoothing)
DPI          = 300
OUTPUT       = "cover_art.png"
BG           = "#020a17"

# Camera & zoom
VIEW_ELEV  = 20          # degrees above horizon
VIEW_AZIM  = -52         # compass: looking NE from SW
XLIM       = (5, 72)     # km — zoomed x window
YLIM       = (5, 33)     # km — zoomed y window

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_dir(directory, suffix=None):
    d = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".npy"):
            continue
        if suffix and not fname.endswith(suffix):
            continue
        try:
            cid = fname.split("_")[1]
        except (IndexError, ValueError):
            continue
        arr = np.load(os.path.join(directory, fname))[::-1, :]
        d.setdefault(cid, []).append(arr)
    return d

def _merge(rasters, fn=np.nanmean):
    with np.errstate(invalid="ignore"):
        return fn(np.stack(rasters, 0), axis=0)

def _fill_nans_nn(arr):
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()
    out = arr.copy()
    _, idx = distance_transform_edt(mask, return_indices=True)
    out[mask] = arr[idx[0][mask], idx[1][mask]]
    return out

# ── Load all data layers ──────────────────────────────────────────────────────
print("Loading bathymetry …", flush=True)
bath = cm_mod.CostMap(dx=DX, dy=DY, default_cost=np.nan)
for cid, rs in _load_dir(DESTRIPED).items():
    avg = _merge(rs, np.nanmean)
    rs5 = ski_measure.block_reduce(avg, block_size=5, func=np.nanmean)
    xs, xe, ys, ye = bath.slice_cost_map(int(cid))
    bath.add_cost(xs, ys, cost=rs5, x_idx_end=xe, y_idx_end=ye)

print("Loading sandwave labels …", flush=True)
sw_cm = cm_mod.CostMap(dx=DX, dy=DY, default_cost=0.0)
for cid, rs in _load_dir(LABELS_DIR, suffix="destriped_labels_smoothed.npy").items():
    rs = [r.astype(float) for r in rs]
    for r in rs:
        r[r == -1] = np.nan
    avg = _merge(rs, np.nanmean)
    rs5 = ski_measure.block_reduce(avg, block_size=5, func=np.max)
    xs, xe, ys, ye = sw_cm.slice_cost_map(int(cid))
    sw_cm.add_cost(xs, ys, cost=rs5, x_idx_end=xe, y_idx_end=ye)

print("Loading variance …", flush=True)
var_cm = cm_mod.CostMap(dx=DX, dy=DY, default_cost=np.nan)
for cid, rs in _load_dir(VAR_DIR).items():
    avg = _merge(rs, np.nanmean)
    rs5 = ski_measure.block_reduce(avg, block_size=5, func=np.nanmean)
    t   = np.log10(np.sqrt(np.maximum(rs5, 0.0)) + 1.0)
    xs, xe, ys, ye = var_cm.slice_cost_map(int(cid))
    var_cm.add_cost(xs, ys, cost=t, x_idx_end=xe, y_idx_end=ye)

print("Loading amplitude …", flush=True)
amp_cm = cm_mod.CostMap(dx=DX, dy=DY, default_cost=np.nan)
for cid, rs in _load_dir(AMP_DIR).items():
    avg = _merge(rs, np.nanmax)
    rs5 = ski_measure.block_reduce(avg, block_size=5, func=np.nanmax)
    xs, xe, ys, ye = amp_cm.slice_cost_map(int(cid))
    amp_cm.add_cost(xs, ys, cost=rs5, x_idx_end=xe, y_idx_end=ye)

# ── Build routing cost grid ───────────────────────────────────────────────────
print("Building routing grid …", flush=True)
var_bin = np.where((~np.isnan(var_cm.costs)) & (var_cm.costs > VAR_THRESH), 1.0, np.nan)
amp_bin = np.where((~np.isnan(amp_cm.costs)) & (amp_cm.costs > AMP_THRESH), 1.0, np.nan)

routing = cm_mod.CostMap(dx=DX, dy=DY, default_cost=1.0)
routing.add_cost(None, None, cost=sw_cm.costs)
routing.add_cost(None, None, cost=var_bin)
routing.add_cost(None, None, cost=amp_bin)
routing.set_nans(bath)
grid = routing.costs

# ── Run A* ────────────────────────────────────────────────────────────────────
print("Running A* …", flush=True)
sx, sy = bath.start_utm
ex, ey = bath.end_utm
sx_idx, sy_idx = bath.get_idx_from_coordinates(sx, sy)
ex_idx, ey_idx = bath.get_idx_from_coordinates(ex, ey)

def _nearest_valid(g, row, col):
    valid = np.isfinite(g) & (g > 0)
    if valid[row, col]:
        return row, col
    vr, vc = np.where(valid)
    dists   = (vr - row) ** 2 + (vc - col) ** 2
    i       = int(np.argmin(dists))
    return int(vr[i]), int(vc[i])

sr, sc = _nearest_valid(grid, sy_idx, sx_idx)
er, ec = _nearest_valid(grid, ey_idx, ex_idx)
print(f"  start ({sy_idx},{sx_idx})->{sr},{sc}  end ({ey_idx},{ex_idx})->{er},{ec}", flush=True)

planner = AStarPlanner(cost_grid=grid, max_turn_steps=1,
                       heuristic_weight=1.0, momentum=MOMENTUM)
astar_result = planner.solve(start=(sr, sc), goal=(er, ec),
                              start_heading=None, goal_heading=None)

if astar_result:
    print(f"  A* found route: {len(astar_result.coords)} steps, "
          f"cost {astar_result.total_cost:.1f}", flush=True)
else:
    print("  A* found no path — continuing without route", flush=True)

# ── Pre-process bathymetry for 3D rendering ───────────────────────────────────
depth    = bath.costs                          # (350, 800) m, positive = below sea
sw_prob  = np.clip(np.nan_to_num(sw_cm.costs), 0.0, 1.0)

nan_mask = np.isnan(depth)
depth_f  = _fill_nans_nn(depth)
depth_s  = ndimage.gaussian_filter(depth_f, sigma=SMOOTH_SIGMA)

# Downsample
Z    = depth_s [::STRIDE, ::STRIDE]
SW   = sw_prob [::STRIDE, ::STRIDE]
NAN  = nan_mask[::STRIDE, ::STRIDE]
rows, cols = Z.shape

z_lo, z_hi = np.nanpercentile(Z, [1, 99])
print(f"Grid {rows}x{cols}  depth p1-p99: {z_lo:.1f}-{z_hi:.1f} m", flush=True)

# Coordinate meshes (km relative to BL)
x_km = np.arange(cols) * DX * STRIDE / 1e3
y_km = np.arange(rows) * DX * STRIDE / 1e3
Xg, Yg = np.meshgrid(x_km, y_km)

# Surface elevation: shallow = high, deep = low (inverted depth)
Z_surf = (z_hi - np.clip(Z, z_lo, z_hi)) * VE / 1e3   # km units

# ── Colour computation ────────────────────────────────────────────────────────
_palette = [
    (0.03, 0.07, 0.22),
    (0.06, 0.16, 0.42),
    (0.09, 0.28, 0.58),
    (0.18, 0.44, 0.65),
    (0.30, 0.58, 0.67),
    (0.50, 0.70, 0.68),
    (0.70, 0.75, 0.62),
    (0.87, 0.80, 0.50),
]
cmap_sea = LinearSegmentedColormap.from_list("seabed", _palette, N=512)

Zn = np.clip((z_hi - Z) / (z_hi - z_lo), 0.0, 1.0)

ls       = LightSource(azdeg=310, altdeg=42)
rgb_base = cmap_sea(Zn)[:, :, :3]
rgb_hs   = ls.shade_rgb(rgb_base, elevation=Z_surf, vert_exag=1.0, blend_mode="overlay")

# Sand wave amber overlay
amber  = np.array([0.98, 0.58, 0.12])
sw_w   = np.clip(SW * 0.45, 0.0, 1.0)[:, :, np.newaxis]
rgb_sw = rgb_hs * (1 - sw_w) + amber * sw_w

# Atmospheric haze (decreasing toward viewer = low y, more toward far edge = high y)
haze_c = np.array([0.06, 0.14, 0.34])
haze_w = ((Yg / Yg.max()) ** 1.5 * 0.18)[:, :, np.newaxis]
rgb_hz = rgb_sw * (1 - haze_w) + haze_c * haze_w

# Gentle no-data tint (seamless gaps)
nd_c = np.array([0.05, 0.12, 0.28])
nd_w = (NAN.astype(float) * 0.45)[:, :, np.newaxis]
rgb  = np.clip(rgb_hz * (1 - nd_w) + nd_c * nd_w, 0.0, 1.0)

fc = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=-1)

# ── Figure ────────────────────────────────────────────────────────────────────
print("Rendering surface …", flush=True)
fig = plt.figure(figsize=(28, 14), facecolor=BG)
ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d", facecolor=BG)

ax.plot_surface(
    Xg, Yg, Z_surf,
    facecolors=fc,
    shade=False,
    rstride=1, cstride=1,
    antialiased=False,
    linewidth=0,
)

# ── A* pipeline route ─────────────────────────────────────────────────────────
z_span   = (z_hi - z_lo) * VE / 1e3
elev_off = z_span * 0.10    # float above seabed

if astar_result:
    coords = np.array(astar_result.coords)   # (N, 2): (row, col) in full 800x350 grid
    xr = coords[:, 1] * DX / 1e3            # col → km
    yr = coords[:, 0] * DX / 1e3            # row → km
    # Z lookup in downsampled grid
    xi = np.clip((xr / (DX * STRIDE / 1e3)).astype(int), 0, cols - 1)
    yi = np.clip((yr / (DX * STRIDE / 1e3)).astype(int), 0, rows - 1)
    zr = Z_surf[yi, xi] + elev_off
    # Glow: outer halo → bright core (white-cyan)
    route_col = "#e0f8ff"   # near-white cyan
    for lw, al in [(20, 0.03), (12, 0.12), (6, 0.38), (2.5, 0.95)]:
        ax.plot(xr, yr, zr, color=route_col, lw=lw, alpha=al,
                solid_capstyle="round", solid_joinstyle="round", zorder=10)

# ── Camera & framing ──────────────────────────────────────────────────────────
ax.set_axis_off()
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

xw = XLIM[1] - XLIM[0]
yw = YLIM[1] - YLIM[0]
ax.set_box_aspect([xw, yw, z_span * 5.0])
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_zlim(-z_span * 0.05, z_span * 1.1)

for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("none")
ax.grid(False)

# ── Save & post-process ───────────────────────────────────────────────────────
TMP = OUTPUT.replace(".png", "_raw.png")
print(f"Saving raw render …", flush=True)
fig.savefig(TMP, dpi=DPI, bbox_inches="tight",
            facecolor=BG, pad_inches=0.05)
plt.close(fig)

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
print("Post-processing …", flush=True)
img = Image.open(TMP).convert("RGB")

# Subtle additive glow (keeps the dark background dark)
img_np  = np.array(img).astype(float)
blur_np = np.array(img.filter(ImageFilter.GaussianBlur(radius=7))).astype(float)
glow    = np.clip(img_np + blur_np * 0.10, 0, 255).astype(np.uint8)
img     = Image.fromarray(glow)

# Colour boost
img = ImageEnhance.Color(img).enhance(1.22)

# Radial vignette centred slightly below middle (where terrain sits)
w, h = img.size
mask = Image.new("L", (w, h), 255)
draw = ImageDraw.Draw(mask)
cx   = w // 2
cy   = int(h * 0.58)   # shift centre slightly downward toward terrain
rx   = int(w * 0.52)
ry   = int(h * 0.52)
draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(radius=int(min(w, h) * 0.22)))
dark = Image.new("RGB", (w, h), (2, 10, 23))
img  = Image.composite(img, dark, mask)

img.save(OUTPUT, dpi=(DPI, DPI))
os.remove(TMP)
print(f"Done: {OUTPUT}")
