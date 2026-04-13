from matplotlib import pyplot as plt

import costmap as cm
from Astar import AStarPlanner
import skimage as ski
import os
import numpy as np

rasters_path = "sandwave_detection_v8/labels"
sand_wave_ratio = 4

def run():
    
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
        costmap.set_cost(x_start, y_start, cost=np.ones((y_end - y_start, x_end - x_start)), x_idx_end=x_end, y_idx_end=y_end)
        costmap.add_cost(x_start, y_start, cost=rescaled_raster*sand_wave_ratio, x_idx_end=x_end, y_idx_end=y_end)
        # costmap.multiply_cost(x_start, y_start, factor=rescaled_raster, x_idx_end=x_end, y_idx_end=y_end)
    
    costmap.block_n2000()
    costmap.plot_cost_map(cmap='viridis', show=True, show_routes=True)
    costmap.fill_nans_nn()
    costmap.plot_cost_map(cmap='viridis', show=True, show_routes=True)
    rescale_factor = 1
    cmap_rescaled = ski.measure.block_reduce(
        costmap.costs,
        block_size=rescale_factor,
        func=np.nanmax
    )
    
    print(f"cmap has size {cmap_rescaled.shape} after rescaling with factor {rescale_factor}")
    planner = AStarPlanner(
        cost_grid=cmap_rescaled,
        max_turn_steps=1,
        heuristic_weight=1.0
    )
    start_x, start_y = costmap.start_utm
    end_x, end_y = costmap.end_utm
    start_x_idx = costmap.get_idx_from_coordinates(start_x, start_y)[0] // rescale_factor
    start_y_idx = costmap.get_idx_from_coordinates(start_x, start_y)[1] // rescale_factor
    end_x_idx = costmap.get_idx_from_coordinates(end_x, end_y)[0] // rescale_factor
    end_y_idx = costmap.get_idx_from_coordinates(end_x, end_y)[1] // rescale_factor
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(start_x_idx, start_y_idx, c='red', marker='X', label='Start')
    ax.scatter(end_x_idx, end_y_idx, c='blue', marker='X', label='End')
    costmap.plot_cost_map(raster=cmap_rescaled, cmap='viridis', show=False, ax=ax)
    ax.legend()
    plt.show()
    
    result, record = planner.solve_with_recording(
        start=(start_y_idx, start_x_idx),  # Note the order (y, x) for grid indexing
        goal=(end_y_idx, end_x_idx)
    )

    if result is not None:
        print(result)
        print(f"Nodes expanded: {len(record.expansions)}")
        record.animate(step=2000, interval=0, route=costmap.routes[1])        
        
def combine_rasters(rasters, func=np.nanmean):
    """Combine multiple rasters using a specified function while ignoring NaN values."""
    stacked = np.stack(rasters, axis=0)
    with np.errstate(invalid='ignore'):
        combined = func(stacked, axis=0)
    return combined
    
if __name__ == "__main__":
    run()