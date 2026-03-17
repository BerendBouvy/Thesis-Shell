from dataloader import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import aoi
import os
import tqdm
from shapely.geometry import Polygon
import time 
from destripeClass import Destriper
import cmocean


def analyse(cell_corners, save_path=None):
    """Run cell-wise destriping and comparison workflow.

    Plot outputs are written to `save_path` when provided (default: `results`).
    """
    dls = pickle.load(open("data_loaders_v2.pkl", "rb"))
    # dls = pickle.load(open("data_loaders_small.pkl", "rb"))
    # dls = pickle.load(open("used_dls.pickle", "rb"))
    print(f"Loaded {len(dls)} data loaders from pickle.")

    angle_dict = {}
    true_angles = pickle.load(open("true_angles.pickle", "rb"))
    
    results_dir = save_path or "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists("destriped_rasters"):
        os.makedirs("destriped_rasters")
    if not os.path.exists("rasters"):
        os.makedirs("rasters")
    if not os.path.exists("destriping_results"):
        os.makedirs("destriping_results")
    
    for cid, (x, y) in cell_corners.items():
        # cid, (x, y) = 42, cell_corners[42]
        cell_results_dir = os.path.join(results_dir, f"cell_{cid}")
        if not os.path.exists(cell_results_dir):
            os.makedirs(cell_results_dir)
        print(f"Processing cell ID: {cid}")
        poly = Polygon([(x, y), (x + 5000, y), (x + 5000, y + 5000), (x, y + 5000)])
        raster_list = []
        raster_destriped_list = []
        dl_list = []
        for dl in dls:
            convex_hull_utm = dl.convex_hull_utm
            if convex_hull_utm.intersects(poly):
                raster, _ = dl.get_raster(location=(x, y), width=5000, height=5000, cell_size=20, point_location='lower_left')
                if raster is None:
                    continue
                elif np.sum(~np.isnan(raster)) < 0.2 * raster.size:
                    print(f"Warning: Only {np.sum(~np.isnan(raster))}/{raster.size} pixels have data in cell {cid} for CDI ID: {dl.metadata['CDI-record id']}. Skipping destriping.")
                    continue
                else:
                    raster_list.append(raster)
                    destriper = Destriper(
                        trend_param=6,
                        style='line',
                        width=5,
                        # pad_style='wrap',
                        pad_style='constant',
                        detrend='gaussian',
                        save_plot=f"destriping_results/cell_{cid}/CDI_{dl.metadata['CDI-record id']}"
                    )
                    destriper.set_angle(true_angles.get(dl.metadata['CDI-record id'], None))  # Use true angle if available, else default to automatic detection
                    destriped = destriper.process(
                        raster, 
                        plot=True)
                    raster_destriped_list.append(destriped)
                    dl_list.append(dl)
                    
                    angle_dict = update_angle_dictionary(angle_dict, cid, dl.metadata['CDI-record id'], destriper.angle)
                    # save rasters
                    np.save(f"rasters/cell_{cid}_CDI_{dl.metadata['CDI-record id']}.npy", raster)
                    np.save(f"destriped_rasters/cell_{cid}_CDI_{dl.metadata['CDI-record id']}_destriped.npy", destriped)
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_title(f"CDI ID: {dl.metadata['CDI-record id']}\nYear: {str(dl.metadata.get('Start Date'))[:4]}")
                    
                    # Set extent to map raster pixels to UTM coordinates
                    raster_extent = [x, x + 5000, y, y + 5000]
                    im = ax.imshow(destriped, cmap=cmocean.cm.deep, extent=raster_extent, origin='upper', aspect='auto')
                    
                    ax.set_xlabel('Easting (m)')
                    ax.set_ylabel('Northing (m)')
                    plt.colorbar(im, ax=ax, label='Mean (m)')
                    plt.savefig(os.path.join(cell_results_dir, f"CDI_{dl.metadata['CDI-record id']}.png"), dpi=300)
                    plt.close(fig)
                    
        for i in range(len(raster_list)):
            for j in range(len(raster_list)):
                if str(dl_list[i].metadata['Start Date']) > str(dl_list[j].metadata['Start Date']):
                    compare_path = os.path.join(
                        cell_results_dir,
                        f"difference_map_{str(dl_list[i].metadata['Start Date'])[:4]}_{str(dl_list[j].metadata['Start Date'])[:4]}.png",
                    )
                    compare_rasters(
                        raster_destriped_list[i],
                        raster_destriped_list[j],
                        dl_list[i],
                        dl_list[j],
                        cid,
                        save_path=compare_path,
                    )
        # break
        
    
    print("\n--- Analysis complete ---")
    print(f"Angle dictionary:\n\n {angle_dict}")
    # Save angle dictionary to a text file
    with open(os.path.join(results_dir, "angle_dictionary2.pickle"), "wb") as f:
        pickle.dump(angle_dict, f)    
    with open("used_dls.pickle", "wb") as f:
        pickle.dump(dl_list, f)
    return 

def update_angle_dictionary(angle_dict, cid, cdi, angle):
    """Store angle for each CDI-cell combination."""
    if angle_dict.get(cdi) is None:
        angle_dict[cdi] = []
    # Append this cell's angle (allows same CDI in multiple cells)
    angle_dict[cdi].append({'cell_id': cid, 'angle': angle})
    return angle_dict   

def plot_cells(aoi_obj, cell_corners, cell_size=5000, save_path=None):
    """Plot AOI with analysis grid cell outlines and IDs.

    If `save_path` is provided, the figure is saved and not shown.
    """
    fig, ax = aoi_obj.plot_aoi()
    for cid, (x, y) in cell_corners.items():
        pol = Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
        gdf = gpd.GeoDataFrame(geometry=[pol], crs=aoi_obj.aoi_gdf.crs)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf_wgs84.plot(ax=ax, edgecolor='blue', linewidth=0.5, facecolor='none')
        
        # Convert text position to WGS84
        center_point = gdf_wgs84.geometry.iloc[0].centroid
        ax.text(center_point.x, center_point.y, str(cid), color='black', fontsize=8, ha='center', va='center')    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
def get_cell_corners(aoi_obj, cell_size=5000):
    width, height = 80e3, 35e3  # in meters
    bottom_left = aoi_obj.aoi_gdf.total_bounds[[0, 1]]  # minx, miny
    
    # dividing the aoi in 5x5km cells
    cell_size = cell_size  # 5 km
    x_starts = np.arange(bottom_left[0], bottom_left[0] + width, cell_size)
    y_starts = np.arange(bottom_left[1], bottom_left[1] + height, cell_size)
    
    # numbering cells from left to right, bottom to top and storing the bottom left corner of each cell
    cell_corners = {}
    cell_id = 0
    for y in y_starts:
        for x in x_starts:
            cell_corners[cell_id] = (x, y)
            cell_id += 1
    return cell_corners

def compare_rasters(raster1, raster2, dl1, dl2, cell_id, save_path=None):
    """Plot and save difference map between two rasters.

    If `save_path` is provided, it is used directly; otherwise a default path is used.
    """
    dif = raster1 - raster2
    if sum(~np.isnan(dif.flatten())) < .1*raster1.size:
        return
    plt.figure(figsize=(6, 6))
    plt.title(f"Difference Map {str(dl1.metadata['Start Date'])[:4]} - {str(dl2.metadata['Start Date'])[:4]}")
    minmax = np.max([-np.nanmin(dif), np.nanmax(dif)])
    plt.imshow(dif, cmap='seismic', vmin=-minmax, vmax=minmax)
    plt.colorbar(label='Difference Value')
    target = save_path or f"results/cell_{cell_id}/difference_map_{str(dl1.metadata['Start Date'])[:4]}_{str(dl2.metadata['Start Date'])[:4]}.png"
    plt.savefig(target, dpi=300)
    plt.close()

def test():
    aoi1 = aoi.AreaOfInterest("AOI.txt")
    # aoi1.plot_aoi()
    dimensions = aoi1.get_dimensions()
    print(f"bottom left corner dimensions (width x height): {aoi1.aoi_gdf.total_bounds[[0,1]]}")
       
    return


if __name__ == "__main__":
    start = time.time()
    aoi1 = aoi.AreaOfInterest("AOI.txt")
    cell_corners = get_cell_corners(aoi1, cell_size=5000)
    # plot_cells(aoi1, cell_corners, cell_size=5000, save_path="plots/aoi_cells.png")
    analyse(cell_corners=cell_corners)
    # test()
    end = time.time()
    print(f"Total execution time: {(end - start)/60:.2f} minutes")