from dataloader import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import aoi
import os
import tqdm
from shapely.geometry import Polygon
import time     


def analyse(cell_corners):
    dls = pickle.load(open("data_loaders_v2.pkl", "rb"))
    print(f"Loaded {len(dls)} data loaders from pickle.")
    # UTM zone 31N
    crs = "EPSG:32631"
    if not os.path.exists("results"):
        os.makedirs("results")
    for cid, (x, y) in cell_corners.items():
        # cid, (x, y) = 42, cell_corners[42]
        if not os.path.exists(f"results/cell_{cid}"):
            os.makedirs(f"results/cell_{cid}")
        print(f"Processing cell ID: {cid}")
        poly = Polygon([(x, y), (x + 5000, y), (x + 5000, y + 5000), (x, y + 5000)])
        cell_gdf = gpd.GeoDataFrame(geometry=[poly], crs=crs)
        raster_list = []
        dl_list = []
        for dl in tqdm.tqdm(dls):
            convex_hull_utm = dl.convex_hull_utm
            if convex_hull_utm.intersects(poly):
                print(f"  Cell {cid} intersects with CDI ID: {dl.metadata['CDI-record id']}")
                raster, _ = dl.get_raster(location=(x, y), width=5000, height=5000, cell_size=30, point_location='lower_left')
                if raster is None:
                    print(f"    No data points in cell {cid} for CDI ID: {dl.metadata['CDI-record id']}")
                    continue
                else:
                    raster_list.append(raster)
                    dl_list.append(dl)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.set_title(f"CDI ID: {dl.metadata['CDI-record id']}\nYear: {str(dl.metadata.get('Start Date'))[:4]}")
                    
                    # Set extent to map raster pixels to UTM coordinates
                    raster_extent = [x, x + 5000, y, y + 5000]
                    im = ax.imshow(raster, cmap='viridis', extent=raster_extent, origin='upper', aspect='auto')
                    
                    # Plot cell boundary in UTM coordinates
                    cell_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
                    
                    ax.set_xlabel('Easting (m)')
                    ax.set_ylabel('Northing (m)')
                    plt.colorbar(im, ax=ax, label='Mean (m)')
                    plt.savefig(f"results/cell_{cid}/CDI_{dl.metadata['CDI-record id']}.png", dpi=300)
                    plt.close(fig)
        for i in range(len(raster_list)):
            for j in range(len(raster_list)):
                if str(dl_list[i].metadata['Start Date']) > str(dl_list[j].metadata['Start Date']):
                    compare_rasters(raster_list[i], raster_list[j], dl_list[i], dl_list[j], cid)
        # break  # remove this break to process all cells
    

def plot_cells(aoi_obj, cell_corners, cell_size=5000, save_path=None):
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

def compare_rasters(raster1, raster2, dl1, dl2, cell_id):
    dif = raster1 - raster2
    if sum(~np.isnan(dif.flatten())) < .1*raster1.size:
        return
    plt.figure(figsize=(6, 6))
    plt.title(f"Difference Map {str(dl1.metadata['Start Date'])[:4]} - {str(dl2.metadata['Start Date'])[:4]}")
    minmax = np.max([-np.nanmin(dif), np.nanmax(dif)])
    plt.imshow(dif, cmap='seismic', vmin=-minmax, vmax=minmax)
    plt.colorbar(label='Difference Value')
    plt.savefig(f"results/cell_{cell_id}/difference_map_{str(dl1.metadata['Start Date'])[:4]}_{str(dl2.metadata['Start Date'])[:4]}.png", dpi=300)
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
    print(f"Time to generate cell corners: {time.time() - start} seconds")
    # plot_cells(aoi1, cell_corners, cell_size=5000, save_path="plots/aoi_cells.png")
    analyse(cell_corners=cell_corners)
    # test()
    end = time.time()
    print(f"Total execution time: {end - start} seconds")