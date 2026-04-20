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
    nanmap.plot_cost_map(cmap=cmocean.cm.deep, show=True, show_routes=False, save_path="temp/nan_map.png")  
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
    costmap.plot_cost_map(cmap='Blues', show=False, show_routes=True, save_path="temp/cost_map_before_filling.png")
    
    # costmap.fill_nans_nn()
    costmap.fill_nans_high_cost()
    costmap.plot_cost_map(cmap='Blues', show=True, show_routes=True, save_path="cmap.png")
    
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
    sigma = .1
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

        combined = cm.nansum([base, costs1, costs2, costs3], axis=0)
            
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
    end = time.time()
    print(f"Execution time: {(end - start) / 60:.2f} minutes")