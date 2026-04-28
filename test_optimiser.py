from matplotlib import pyplot as plt
import cmocean
import costmap as cm
from Astar import AStarPlanner
import skimage as ski
import os
import numpy as np
import time

rasters_path = "sandwave_detection_v8/labels"
animate = False
n2000 = False
if n2000:
    ref_plot = 1
else:   
    ref_plot = 0

def nan_map():
    costmap = cm.CostMap(dx=100, dy=100, default_cost=np.nan)
    files = os.listdir("destriped_rasters")
    raster_dict = {}
    for file in files:
        id = file.split('_')[1]
        raster = np.load(os.path.join("destriped_rasters", file))[::-1, :]
        if id not in raster_dict:
            raster_dict[id] = []
        raster_dict[id].append(raster)
        
    for id, rasters in raster_dict.items():
        avg_raster = combine_rasters(rasters, func=np.nanmean)
        rescaled_raster = ski.measure.block_reduce(
            avg_raster,
            block_size=5,
            func=np.nanmean
        )
        x_start, x_end, y_start, y_end = costmap.slice_cost_map(int(id))
        costmap.add_cost(x_start, y_start, cost=rescaled_raster, x_idx_end=x_end, y_idx_end=y_end)
    return costmap

def build_sandwave_costmap():
    costmap = cm.CostMap(dx=100, dy=100, default_cost=np.nan)
    files = [f for f in os.listdir(rasters_path) if f.endswith("destriped_labels_smoothed.npy")]
    raster_dict = {}
    for file in files:
        id = file.split('_')[1]
        raster = np.load(os.path.join(rasters_path, file))[::-1, :]
        raster = raster.astype(float)
        raster[raster==-1] = np.nan
        if id not in raster_dict:
            raster_dict[id] = []
        raster_dict[id].append(raster)

    for id, rasters in raster_dict.items():
        avg_raster = combine_rasters(rasters, func=np.nanmean)
        
        rescaled_raster = ski.measure.block_reduce(
            avg_raster,
            block_size=5,
            func=np.max
        )
        x_start, x_end, y_start, y_end = costmap.slice_cost_map(int(id))
        
        costmap.add_cost(x_start, y_start, cost=rescaled_raster, x_idx_end=x_end, y_idx_end=y_end)
    return costmap


def build_variance_costmap():
    costmap2 = cm.CostMap(dx=100, dy=100, default_cost=np.nan)
    files2 = "variance_rasters/Rasters"
    raster_dict2 = {}
    for file in os.listdir(files2):
        id2 = file.split('_')[1]
        raster2 = np.load(os.path.join(files2, file))[::-1, :]
        if id2 not in raster_dict2:
            raster_dict2[id2] = []
        raster_dict2[id2].append(raster2)

    for id, rasters in raster_dict2.items():
        avg_raster = combine_rasters(rasters, func=np.nanmean)
        rescaled_raster = ski.measure.block_reduce(
            avg_raster,
            block_size=5,
            func=np.nanmean
        )
        x_start, x_end, y_start, y_end = costmap2.slice_cost_map(int(id))
        # costmap2.set_cost(x_start, y_start, cost=np.ones((y_end - y_start, x_end - x_start)), x_idx_end=x_end, y_idx_end=y_end)
        costmap2.add_cost(x_start, y_start, cost=np.log10(np.sqrt(rescaled_raster)+1), x_idx_end=x_end, y_idx_end=y_end)
    return costmap2

def build_amplitude_costmap():
    costmap3 = cm.CostMap(dx=100, dy=100, default_cost=np.nan)
    files3 = "amplitude_rasters/Rasters_amp"
    raster_dict3 = {}
    for file in os.listdir(files3):
        id3 = file.split('_')[1]
        raster3 = np.load(os.path.join(files3, file))[::-1, :]
        if id3 not in raster_dict3:
            raster_dict3[id3] = []
        raster_dict3[id3].append(raster3)
    
    for id, rasters in raster_dict3.items():
        avg_raster = combine_rasters(rasters, func=np.nanmax)
        rescaled_raster = ski.measure.block_reduce(
            avg_raster,
            block_size=5,
            func=np.nanmax
        )
        x_start, x_end, y_start, y_end = costmap3.slice_cost_map(int(id))
        costmap3.add_cost(x_start, y_start, cost=rescaled_raster, x_idx_end=x_end, y_idx_end=y_end)
    return costmap3


def run():
    costmap = cm.CostMap(dx=100, dy=100, default_cost=1)
    nanmap = nan_map()
    # nanmap.plot_cost_map(cmap=cmocean.cm.deep, show=True, show_routes=False, save_path="temp/nan_map.png")  
    costmap1 = build_sandwave_costmap()
    costmap2 = build_variance_costmap()
    
    var_threshold = 0.05
    costmap2.set_cost(x_idx=None, y_idx=None, cost=np.where(costmap2.costs > var_threshold, 1, np.nan))
    costmap3 = build_amplitude_costmap()
    amp_threshold = 0.08
    costmap3.set_cost(x_idx=None, y_idx=None, cost=np.where(costmap3.costs > amp_threshold, 1, np.nan))

    # costmap1.plot_cost_map(cmap='Reds', show=True, show_routes=True, save_path="temp/sandwave_cost_map.png")
    # costmap2.plot_cost_map(cmap='Reds', show=True, show_routes=True, save_path="temp/variance_cost_map.png")
    # costmap3.plot_cost_map(cmap='Reds', show=True, show_routes=True)    
    
    costmap.add_cost(x_idx=None, y_idx=None, cost=costmap1.costs)
    costmap.add_cost(x_idx=None, y_idx=None, cost=costmap2.costs)
    costmap.add_cost(x_idx=None, y_idx=None, cost=costmap3.costs)
    
    costmap.set_nans(nanmap)

    if n2000:
        costmap.block_n2000()
    # costmap.plot_cost_map(cmap='Blues', show=False, show_routes=True, save_path="temp/cost_map_before_filling.png")
    
    costmap.plot_cost_map(cmap='Reds', show=True, show_routes=True)
    costmap.fill_nans_nn(max_gap=1000)
    costmap.plot_cost_map(cmap='Greens', show=True, show_routes=True)
    costmap.fill_nans_high_cost()
    costmap.plot_cost_map(cmap='Blues', show=True, show_routes=True, save_path="cmap.png")
    return
    rescale_factor = 1
    # cmap_rescaled = ski.measure.block_reduce(
    #     costmap.costs,
    #     block_size=rescale_factor,
    #     func=np.nanmax
    # )
    
    planner = AStarPlanner(
        cost_grid=costmap.costs,
        max_turn_steps=1,
        heuristic_weight=1.0,
        momentum=8
    )
    start_x, start_y = costmap.start_utm
    end_x, end_y = costmap.end_utm
    
    start_x_idx = costmap.get_idx_from_coordinates(start_x, start_y)[0] // rescale_factor
    start_y_idx = costmap.get_idx_from_coordinates(start_x, start_y)[1] // rescale_factor
    end_x_idx = costmap.get_idx_from_coordinates(end_x, end_y)[0] // rescale_factor
    end_y_idx = costmap.get_idx_from_coordinates(end_x, end_y)[1] // rescale_factor

    result, record = planner.solve_with_recording(
        start=(start_y_idx, start_x_idx),  # Note the order (y, x) for grid indexing
        goal=(end_y_idx, end_x_idx),
        start_heading=0,
        goal_heading=6
    )

    if result is not None:
        print(result)
        print(f"Nodes expanded: {len(record.expansions)}")
        if animate:
            record.animate(step=2000, interval=100, route=costmap.routes[ref_plot], save_gif="gifs/astar_pathfinding.gif") 
        else:
            result.plot_path(show=True, route=costmap.routes[ref_plot])

def run_sensitivity_analysis():
    n = 100
    sigma = .5
    var_threshold = 0.05
    amp_threshold = 0.08
    rescale_factor = 4
    momentum = 2

    # Build all data sources once
    nanmap = nan_map()
    base1 = build_sandwave_costmap().costs
    base2_thresh = np.where(build_variance_costmap().costs > var_threshold, 1.0, np.nan)
    base3_thresh = np.where(build_amplitude_costmap().costs > amp_threshold, 1.0, np.nan)
    shape = base1.shape

    # Build template once for metadata: nan mask, n2000, routes, coordinates
    template = cm.CostMap(dx=100, dy=100, default_cost=1)
    # plt.subplot(1, 3, 1)
    # plt.imshow(base1, cmap='Reds', origin='upper')
    # plt.subplot(1, 3, 2)
    # plt.imshow(base2_thresh, cmap='Reds', origin='upper')
    # plt.subplot(1, 3, 3)
    # plt.imshow(base3_thresh, cmap='Reds', origin='upper')
    # plt.show()
    
    template.set_nans(nanmap)
    nan_mask = np.isnan(template.costs)

    start_x, start_y = template.start_utm
    end_x, end_y = template.end_utm
    start_x_idx = template.get_idx_from_coordinates(start_x, start_y)[0] // rescale_factor
    start_y_idx = template.get_idx_from_coordinates(start_x, start_y)[1] // rescale_factor
    end_x_idx = template.get_idx_from_coordinates(end_x, end_y)[0] // rescale_factor
    end_y_idx = template.get_idx_from_coordinates(end_x, end_y)[1] // rescale_factor

    routes = []
    for _ in range(n):
        costs1 = add_noise(base1, sigma)
        costs2 = add_noise(base2_thresh, sigma)
        costs3 = add_noise(base3_thresh, sigma)
        # template.plot_cost_map(raster=costs1, cmap='Reds', show=True)
        # template.plot_cost_map(raster=costs2, cmap='Reds', show=True)
        # template.plot_cost_map(raster=costs3, cmap='Reds', show=True)

        base = add_noise(np.ones(shape), sigma)
        
        # weights = np.random.uniform(0, 2, size=3)
        weights = [1, 1, 1]
        combined = cm.nansum([base, weights[0]*costs1, weights[1]*costs2, weights[2]*costs3], axis=0)
            
        combined[nan_mask] = np.nan
        combined[np.isnan(combined)] = np.nanmax(combined)
        
        if n2000:
            combined = template.block_n2000(array=combined)

        if rescale_factor > 1:
            combined = ski.measure.block_reduce(combined, block_size=rescale_factor, func=np.nanmax)
            rescaled_shape = combined.shape

        planner = AStarPlanner(
            cost_grid=combined,
            max_turn_steps=1,
            heuristic_weight=1,
            momentum=momentum
        )

        result, record = planner.solve_with_recording(
            start=(start_y_idx, start_x_idx),
            goal=(end_y_idx, end_x_idx),
            start_heading=0,
            goal_heading=6
        )

        if result is not None:
            print(result)
            print(f"Nodes expanded: {len(record.expansions)}")
            if animate:
                record.animate(step=100000, interval=100, route=template.routes[ref_plot], save_gif="gifs/astar_pathfinding.gif")
            else:
                routes.append(result)
    
    # create heatmap of route frequencies
    heatmap = np.zeros(rescaled_shape if rescale_factor > 1 else shape)
    for route in routes:
        heatmap += route.get_numpy_path()
    heatmap[heatmap == 0] = np.nan  # Set non-visited cells to NaN for better visualization
    ref_route = template.routes[ref_plot].astype(float)
    if rescale_factor > 1:
        ref_route = ski.measure.block_reduce(ref_route, block_size=rescale_factor, func=np.max)
    ref_route = np.where(ref_route, 1, np.nan)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(combined, cmap=cmocean.cm.deep, alpha=0.5, origin='lower')
    im = ax.imshow(heatmap, cmap='hot', origin='lower', alpha=0.75)
    plt.colorbar(im, ax=ax, label='Route Frequency')
    im = ax.imshow(ref_route, cmap='tab10', origin='lower')

    ax.set_title(f"Route Frequency Heatmap over {n} Runs (sigma={sigma})")
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    plt.savefig(f"temp/route_frequency_heatmap_sigma_{sigma}_{time.time()}.png", dpi=300)
    # plt.show()

    # --- Consensus route: A* on inverse-frequency cost map ---
    # Cells visited by many routes get low cost; unvisited cells get high cost.
    # consensus_cost in range [1, n+1]: visited n times → 1, visited 0 times → n+1.
    heatmap_filled = np.where(np.isnan(heatmap), 0, heatmap)
    consensus_cost = (n + 1) / (heatmap_filled + 1)

    # Re-apply the nan mask (rescaled to match the heatmap resolution)
    if rescale_factor > 1:
        rescaled_nan_mask = ski.measure.block_reduce(
            nan_mask.astype(float), block_size=rescale_factor, func=np.max
        ).astype(bool)
    else:
        rescaled_nan_mask = nan_mask
    consensus_cost[rescaled_nan_mask] = np.nanmax(consensus_cost)

    consensus_planner = AStarPlanner(
        cost_grid=consensus_cost,
        max_turn_steps=1,
        heuristic_weight=1.0,
        momentum=momentum
    )
    consensus_result = consensus_planner.solve(
        start=(start_y_idx, start_x_idx),
        goal=(end_y_idx, end_x_idx),
        start_heading=0,
        goal_heading=6,
    )

    if consensus_result is not None:
        consensus_overlay = np.where(consensus_result.get_numpy_path(), 1.0, np.nan)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(combined, cmap=cmocean.cm.deep, alpha=0.5, origin='lower')
        im2 = ax2.imshow(heatmap, cmap='hot', origin='lower', alpha=0.75)
        plt.colorbar(im2, ax=ax2, label='Route Frequency')
        ax2.imshow(ref_route, cmap='tab10', origin='lower')
        ax2.imshow(consensus_overlay, cmap='cool', origin='lower', alpha=0.9)
        ax2.set_title(f"Consensus Route over {n} Runs (sigma={sigma})")
        ax2.set_xlabel('X Index')
        ax2.set_ylabel('Y Index')
        plt.savefig(f"temp/consensus_route_sigma_{sigma}_{time.time()}.png", dpi=300)
        # plt.show()
    else:
        print("Consensus route: no path found.")
    
def plot_nan_map_with_cells():
    """Plot the nan_map with UTM coords, data-count overlay, routes, and cell index labels."""
    import matplotlib.patheffects as pe
    import rasterio
    import geopandas as gpd

    # Use a throw-away CostMap only for grid geometry (bl, dx, dy, routes)
    costmap = cm.CostMap(dx=100, dy=100, default_cost=np.nan)
    bl = costmap.bl
    dx, dy = costmap.dx, costmap.dy

    fig, ax = plt.subplots(figsize=(10, 5))

    # Base layer: data count raster
    with rasterio.open("data_count_raster.tif") as r:
        data = r.read(1).astype(float)
        if r.nodata is not None:
            data[data == r.nodata] = np.nan
        tif_extent = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
    im_overlay = ax.imshow(
        data,
        cmap='Greens',
        extent=tif_extent,
        origin='upper',
        vmin=0, vmax=4,
    )
    cb = fig.colorbar(im_overlay, ax=ax, label="Number of surveys",
                      fraction=0.03, pad=0.02, shrink=0.7)
    cb.ax.locator_params(nbins=5)

    # Routes
    gdf  = gpd.read_file("shapes/line1.shp").to_crs(costmap.csr)
    gdf2 = gpd.read_file("shapes/line2.shp").to_crs(costmap.csr)
    gdf.plot(ax=ax,  color='red',  linewidth=1.5, label='Route 1')
    gdf2.plot(ax=ax, color='blue', linewidth=1.5, label='Route 2')

    # Cell grid + number labels
    for i in range(112):
        x_start, x_end, y_start, y_end = costmap.slice_cost_map(i)
        cx_utm = bl[0] + (x_start + x_end) / 2 * dx
        cy_utm = bl[1] + (y_start + y_end) / 2 * dy
        ax.text(cx_utm, cy_utm, str(i), ha='center', va='center', fontsize=10,
                color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        rect = plt.Rectangle(
            (bl[0] + x_start * dx, bl[1] + y_start * dy),
            (x_end - x_start) * dx,
            (y_end - y_start) * dy,
            linewidth=0.5, edgecolor='white', facecolor='none', alpha=0.4
        )
        ax.add_patch(rect)

    ax.set_xlim(bl[0], bl[0] + 800 * dx)
    ax.set_ylim(bl[1], bl[1] + 350 * dy)
    ax.set_title("Number of serveys per cell with routes and cell indices")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc='upper left', fontsize='medium', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig("C:\\Users\\beren\\Documents\\AES\\MSc\\Thesis\\Report-Thesis\\figures\\data_count_raster_with_routes.png", dpi=300)
    plt.show()


def plot_turn_rules(momentum=4, staircase_width=3, n_cycles=3):
    """
    Two-panel publish figure for the heading-aware A* kinematic constraints.

    Left  — Rose diagram of legal / illegal next headings from a freshly-turned
             state (heading=NE, s=momentum-1, prev_heading=E).
    Right — Staircase path on a grid showing that prev_heading is NOT
             overwritten on straight steps, so every NE re-entry is a
             compensating turn and therefore always legal regardless of s.
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    # ---- typography ---------------------------------------------------------
    FS       = 14    # panel titles, axis labels, boxed text
    FS_SMALL = 12    # legend, secondary labels
    FS_ANN   = 11    # cell annotations in right panel

    # ---- shared geometry ----------------------------------------------------
    m         = momentum
    DIRS_XY   = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    DIR_NAMES = ['E','NE','N','NW','W','SW','S','SE']

    # ---- colours (consistent between panels) --------------------------------
    C_LEGAL   = '#2ca02c'   # green
    C_ILLEGAL = '#d62728'   # red
    C_GREY    = '#aaaaaa'   # unreachable
    C_BLUE    = 'steelblue'
    C_ORANGE  = 'darkorange'
    C_GREEN   = 'mediumseagreen'
    C_LBLUE   = '#a8d4f5'   # light blue (straight_ph)

    TYPE_COLOR = {
        'straight_init': C_BLUE,
        'first_turn':    C_ORANGE,
        'compensating':  C_GREEN,
        'straight_ph':   C_LBLUE,
    }

    # ---- figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(24, 13))
    gs  = fig.add_gridspec(1, 2, width_ratios=[0.65, 2.0], wspace=0.22)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    fig.subplots_adjust(bottom=0.22, top=0.91)

    # ======================================================================== #
    # LEFT PANEL — turn-rule rose diagram                                       #
    # ======================================================================== #
    cur_heading = 1      # NE
    cur_s       = m - 1
    cur_ph      = 0      # E
    cx, cy      = 0.0, 0.0
    ARROW_LEN   = 1.9
    TEXT_OFF    = 0.40   # extra offset beyond arrow tip

    for h_idx, (adx, ady) in enumerate(DIRS_XY):
        diff = min(abs(cur_heading - h_idx) % 8,
                   (8 - abs(cur_heading - h_idx)) % 8)

        is_straight     = (h_idx == cur_heading)
        is_compensating = (h_idx == cur_ph)
        within_reach    = (diff <= 1)

        if not within_reach:
            color, lw, reason = C_GREY, 1.4, 'unreachable\n(> 45°)'
        elif is_straight or is_compensating:
            color, lw = C_LEGAL, 3.2
            reason = 'straight' if is_straight else 'compensating\n(next_h = prev_h)'
        else:
            color, lw = C_ILLEGAL, 3.2
            reason = 'blocked\n(s > 0,  not compensating)'

        ex, ey = cx + adx * ARROW_LEN, cy + ady * ARROW_LEN
        ax_l.annotate(
            '', xy=(ex, ey), xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=22),
        )
        # label: white background box for readability
        ax_l.text(
            ex + adx * TEXT_OFF, ey + ady * TEXT_OFF,
            f'{DIR_NAMES[h_idx]}\n{reason}',
            ha='center', va='center', fontsize=FS_SMALL,
            color=color, fontweight='bold' if within_reach else 'normal',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.75),
        )

    # Current-state dot
    ax_l.plot(cx, cy, 'ko', ms=14, zorder=6)

    # Incoming arrow (arrived from SW, heading NE)
    ax_l.annotate(
        '', xy=(cx, cy), xytext=(cx - 1.05, cy - 1.05),
        arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=2.6, mutation_scale=20),
    )
    ax_l.text(
        cx - 1.55, cy - 0.80,
        'arrived via NE\n(first turn from E)',
        ha='center', va='top', fontsize=FS_SMALL - 1, color=C_BLUE,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=C_BLUE, linewidth=0.8, alpha=0.90),
    )

    # State box — top
    ax_l.text(
        cx, 3.05,
        f'State:  heading = NE,  s = {cur_s},  prev_heading = E',
        ha='center', va='center', fontsize=FS,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbe6',
                  edgecolor='#c8a800', linewidth=1.2, alpha=0.97),
    )

    # Rule box — bottom
    ax_l.text(
        cx, -3.05,
        'Turn rule:  legal if  s = 0  OR  next_h = prev_heading',
        ha='center', va='center', fontsize=FS,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd',
                  edgecolor='steelblue', linewidth=1.2, alpha=0.97),
    )

    ax_l.set_xlim(-3.4, 3.4)
    ax_l.set_ylim(-3.7, 3.7)
    ax_l.set_aspect('equal')
    ax_l.axis('off')
    ax_l.set_title(
        f'Turn legality diagram  (momentum = {m})\n'
        f'State after first turn:  heading = NE,  s = {cur_s},  prev_heading = E',
        fontsize=FS + 1, pad=14, fontweight='bold',
    )

    legend_l = [
        Line2D([0],[0], color=C_LEGAL,   lw=2.5, label='Legal  (straight or compensating)'),
        Line2D([0],[0], color=C_ILLEGAL, lw=2.5, label='Illegal  (s > 0,  not compensating)'),
        Line2D([0],[0], color=C_GREY,    lw=1.5, label='Unreachable  (> 45° from current heading)'),
    ]
    ax_l.legend(
        handles=legend_l, fontsize=FS_SMALL,
        loc='upper center', bbox_to_anchor=(0.5, -0.01),
        frameon=True, ncol=1, edgecolor='#cccccc',
    )

    # ======================================================================== #
    # RIGHT PANEL — staircase path                                              #
    # ======================================================================== #
    positions, states = [], []
    x, y    = 0, 0
    heading = 0   # E
    s       = 0
    ph      = 0   # E

    # Initial straight East steps (free to turn, s=0)
    for _ in range(m):
        positions.append((x, y))
        states.append({'h': heading, 's': s, 'ph': ph, 'type': 'straight_init'})
        x += 1
        s = max(0, s - 1)

    # First turn E -> NE
    x += 1;  y += 1
    positions.append((x, y))
    states.append({'h': 1, 's': m - 1, 'ph': 0, 'type': 'first_turn'})
    heading, s, ph = 1, m - 1, 0

    # Staircase cycles
    for _ in range(n_cycles):
        for _ in range(staircase_width):
            next_h = 0   # E
            bdx, bdy = DIRS_XY[next_h]
            x += bdx;  y += bdy
            if next_h == heading:          # straight
                new_s, new_ph = max(0, s - 1), ph    # ph UNCHANGED
                stype = 'straight_ph'
            else:                          # compensating (E after NE)
                new_s, new_ph = m - 1, heading
                stype = 'compensating'
            positions.append((x, y))
            states.append({'h': next_h, 's': new_s, 'ph': new_ph, 'type': stype})
            heading, s, ph = next_h, new_s, new_ph

        next_h = 1   # NE (compensating)
        bdx, bdy = DIRS_XY[next_h]
        x += bdx;  y += bdy
        new_s, new_ph = m - 1, heading
        positions.append((x, y))
        states.append({'h': next_h, 's': new_s, 'ph': new_ph, 'type': 'compensating'})
        heading, s, ph = next_h, new_s, new_ph

    # Trim final 5 cells to reduce crowding
    positions = positions[:-5]
    states    = states[:-5]

    xs_all = [p[0] for p in positions]
    ys_all = [p[1] for p in positions]
    x_lo = int(min(xs_all))
    x_hi = int(max(xs_all))
    y_lo = int(min(ys_all))
    y_hi = int(max(ys_all))

    ax_r.set_xlim(x_lo - 0.7, x_hi + 0.7)
    ax_r.set_ylim(y_lo - 0.7, y_hi + 2.8)    # generous top for bracket labels
    ax_r.set_xticks(np.arange(x_lo, x_hi + 1), minor=True)
    ax_r.set_yticks(np.arange(y_lo, y_hi + 1), minor=True)
    ax_r.grid(which='minor', color='#dddddd', linewidth=0.6)
    ax_r.tick_params(which='minor', length=0)
    ax_r.tick_params(axis='both', which='major', labelsize=FS_SMALL)

    # Shade cells
    for (px, py), st in zip(positions, states):
        alpha = 0.65 if st['type'] == 'first_turn' else 0.50
        ax_r.add_patch(plt.Rectangle(
            (px - 0.5, py - 0.5), 1, 1,
            facecolor=TYPE_COLOR[st['type']], alpha=alpha,
            edgecolor='#888888', linewidth=0.7, zorder=2,
        ))

    # Direction arrows between consecutive cells
    for i in range(len(positions) - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        col = TYPE_COLOR[states[i + 1]['type']]
        ax_r.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color=col, lw=2.2, mutation_scale=16),
            zorder=3,
        )

    # State labels inside cells — two-line compact format
    for (px, py), st in zip(positions, states):
        ax_r.text(
            px, py,
            f"h={DIR_NAMES[st['h']]}\ns={st['s']}\nph={DIR_NAMES[st['ph']]}",
            ha='center', va='center', fontsize=FS_ANN,
            linespacing=1.35, zorder=4,
        )

    # "ph unchanged" brackets above each run of straight_ph cells
    i = 0
    while i < len(states):
        if states[i]['type'] == 'straight_ph':
            j = i
            while j < len(states) and states[j]['type'] == 'straight_ph':
                j += 1
            bx0 = positions[i][0]     - 0.38
            bx1 = positions[j - 1][0] + 0.38
            by  = positions[i][1]     + 0.62    # just above cell top (0.5)
            mid = (bx0 + bx1) / 2
            # horizontal span line
            ax_r.annotate(
                '', xy=(bx1, by), xytext=(bx0, by),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.6),
                zorder=5,
            )
            # label above the line
            ax_r.text(
                mid, by + 0.22,
                'ph unchanged  (straight steps)',
                ha='center', va='bottom', fontsize=FS_ANN,
                color='#333333', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.80),
                zorder=6,
            )
            i = j
        else:
            i += 1

    ax_r.set_aspect('equal')
    ax_r.set_xlabel('East  (cells)', fontsize=FS, labelpad=6)
    ax_r.set_ylabel('North  (cells)', fontsize=FS, labelpad=6)
    ax_r.set_title(
        f'Staircase path  (momentum = {m},  {staircase_width} East steps per cycle)\n'
        f'h = heading,   s = straight steps remaining,   ph = prev_heading',
        fontsize=FS + 1, pad=14, fontweight='bold',
    )

    legend_r = [
        mpatches.Patch(facecolor=C_BLUE,   alpha=0.55, label=f'Initial straight East  (s = 0, free to turn)'),
        mpatches.Patch(facecolor=C_ORANGE, alpha=0.75, label=f'First turn E -> NE  (s resets to {m - 1})'),
        mpatches.Patch(facecolor=C_GREEN,  alpha=0.60, label='Compensating turn  (next_h = ph, always legal)'),
        mpatches.Patch(facecolor=C_LBLUE,  alpha=0.60, label='Straight East step  (ph preserved, NE stays compensating)'),
    ]
    ax_r.legend(
        handles=legend_r, fontsize=FS_SMALL,
        loc='upper center', bbox_to_anchor=(0.5, -0.20),
        frameon=True, ncol=2, edgecolor='#cccccc',
    )

    plt.suptitle(
        'A* kinematic constraints: legal vs illegal turns',
        fontsize=FS + 4, fontweight='bold', y=0.97,
    )
    os.makedirs('temp', exist_ok=True)
    plt.savefig(f'temp/turn_rules_momentum_{m}.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_bend_radius(momentum_values=None, cell_size=100):
    """
    Construct the smallest possible octagonal turn path for each momentum value,
    compute its area via the shoelace formula, derive the radius of the
    area-equivalent circle, and plot both the octagon and circle on a cell grid.

    With max_turn_steps=1 (45°/step) and momentum=m, the tightest turn is an
    octagon where each of the 8 sides consists of m steps in one direction.
    The octagon area (in cell units) is exactly 7*m^2, giving radius:
        R = sqrt(7/pi) * m * cell_size  ≈  1.493 * m * cell_size

    Parameters
    ----------
    momentum_values : list of int
    cell_size : float  Cell size in metres (default 100).
    """
    if momentum_values is None:
        momentum_values = [1, 2, 4, 8]

    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches

    # Direction deltas (dx=East, dy=North) for the 8 headings 0-7
    DIRS_XY = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]

    n = len(momentum_values)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    radii = []

    for ax, m in zip(axes, momentum_values):
        # --- Trace path cells (m steps in each of the 8 directions) ----------
        path_cells = []
        x, y = 0, 0
        for dx, dy in DIRS_XY:
            for _ in range(m):
                path_cells.append((x, y))
                x += dx
                y += dy

        # --- Octagon vertices (corners at start of each segment) -------------
        verts = []
        vx, vy = 0, 0
        for dx, dy in DIRS_XY:
            verts.append((float(vx), float(vy)))
            vx += m * dx
            vy += m * dy

        # --- Shoelace area (in cell² units) ----------------------------------
        nv = len(verts)
        area_cells = 0.0
        for i in range(nv):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % nv]
            area_cells += x0 * y1 - x1 * y0
        area_cells = abs(area_cells) / 2.0          # should equal 7*m^2

        area_m2   = area_cells * cell_size ** 2
        radius_m  = np.sqrt(area_m2 / np.pi)
        radii.append(radius_m)

        # Centroid of octagon
        cx = sum(v[0] for v in verts) / nv          # = m/2
        cy = sum(v[1] for v in verts) / nv          # = 3m/2

        # --- Build raster grid -----------------------------------------------
        all_x = [v[0] for v in verts]
        all_y = [v[1] for v in verts]
        pad   = max(2, m // 2 + 1)
        x_min, x_max = int(min(all_x)) - pad, int(max(all_x)) + pad
        y_min, y_max = int(min(all_y)) - pad, int(max(all_y)) + pad
        gw, gh = x_max - x_min + 1, y_max - y_min + 1

        grid = np.zeros((gh, gw))

        # Interior cells (point-in-polygon)
        closed = verts + [verts[0]]
        mpl_poly = MplPath([(v[0], v[1]) for v in closed])
        for gy in range(gh):
            for gx in range(gw):
                wx, wy = gx + x_min, gy + y_min
                if mpl_poly.contains_point((wx, wy)):
                    grid[gy, gx] = 0.4

        # Path cells (drawn on top)
        for (px, py) in path_cells:
            gx = int(px) - x_min
            gy = int(py) - y_min
            if 0 <= gy < gh and 0 <= gx < gw:
                grid[gy, gx] = 1.0

        # --- Plot ------------------------------------------------------------
        ax.imshow(
            grid, origin='lower', cmap='Blues', vmin=0, vmax=1,
            extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5],
        )

        # Minor grid lines at every cell
        ax.set_xticks(np.arange(x_min, x_max + 1), minor=True)
        ax.set_yticks(np.arange(y_min, y_max + 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.3, alpha=0.4)
        ax.tick_params(which='minor', length=0)

        # Octagon outline
        ox = [v[0] for v in verts] + [verts[0][0]]
        oy = [v[1] for v in verts] + [verts[0][1]]
        ax.plot(ox, oy, 'b-', linewidth=1.5)

        # Equivalent circle
        r_cells = radius_m / cell_size
        circ = plt.Circle(
            (cx, cy), r_cells, fill=False,
            color='red', linestyle='--', linewidth=1.5,
        )
        ax.add_patch(circ)
        ax.plot(cx, cy, 'r+', ms=8, markeredgewidth=1.5)

        # Legend
        handles = [
            mpatches.Patch(facecolor='steelblue', label=f'Octagon  ({area_cells:.0f} cells²)'),
            mpatches.Patch(facecolor='lightblue', label='Interior'),
            plt.Line2D([0], [0], color='red', linestyle='--', label=f'Circle  R = {radius_m:.0f} m'),
        ]
        ax.legend(handles=handles, fontsize=11, loc='upper right')

        ax.set_aspect('equal')
        ax.set_title(f'momentum = {m}\nR ≈ {radius_m:.0f} m', fontsize=14)
        ax.set_xlabel('cells (East →)', fontsize=12)
        ax.set_ylabel('cells (North ↑)', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        f'Minimum-turn octagon and area-equivalent circle  (cell size = {cell_size} m)',
        fontsize=16,
    )
    plt.tight_layout()
    os.makedirs('temp', exist_ok=True)
    plt.savefig('temp/bend_radius.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Summary table
    print(f"\n{'momentum':>10}  {'area (cells²)':>15}  {'radius (m)':>12}")
    print('-' * 42)
    for m, r in zip(momentum_values, radii):
        print(f"{m:>10}  {7*m**2:>15.0f}  {r:>12.1f}")

    return dict(zip(momentum_values, radii))


def add_noise(arr, sigma):
    """Add N(0, sigma) noise to all non-NaN, non-zero cells. NaN and 0 cells are unchanged."""
    noisy = arr + np.random.normal(0, sigma, arr.shape)
    return np.where((~np.isnan(arr)) & (arr != 0), noisy, arr)


def combine_rasters(rasters, func=np.nanmean):
    """Combine multiple rasters using a specified function while ignoring NaN values."""
    stacked = np.stack(rasters, axis=0)
    with np.errstate(invalid='ignore'):
        combined = func(stacked, axis=0)
    return combined
    
if __name__ == "__main__":
    start = time.time()
    # run()
    run_sensitivity_analysis()
    # plot_nan_map_with_cells()
    # plot_turn_rules(momentum=4, staircase_width=3, n_cycles=3)
    # plot_bend_radius(momentum_values=[1, 2, 4, 8], cell_size=100)
    end = time.time()
    print(f"Execution time: {(end - start) / 60:.2f} minutes")