import geopandas as gpd
import pyproj as pp
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import numpy as np
import rasterio
import pickle
from dataloader import dataLoader
class AreaOfInterest:
    def __init__(self, file_path, crs="EPSG:32631"):
        # Use UTM Zone 31N as default CRS
        self.file_path = file_path
        self.crs = crs
        self.aoi_gdf = self.load_aoi()
        
    def load_aoi(self):
        # Load AOI from a text file containing coordinates
        # First column x (coordinates), second column y (coordinates)
        coords = []
        with open(self.file_path, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().replace(',', '.').split())
                coords.append((x, y))
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame(index=[0], crs=self.crs, geometry=[polygon])
        return gdf
    
    def plot_aoi(self):
        # Plot the AOI and coastline
        fig, ax = plt.subplots()
        ax = plt.axes(projection=ccrs.PlateCarree())
        self.aoi_gdf.to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
        ax.coastlines(resolution='10m')

        ax.set_title("Area of Interest")
        
        
        plt.show()
        
        
    def get_raster(self, dataloaders, delta_x=1000, delta_y=1000, plot=False, normalize=True):
        # Generate a raster grid within the AOI with specified spacing
        bounds = self.aoi_gdf.total_bounds  # minx, miny, maxx, maxy
        x_min, y_min, x_max, y_max = bounds
        
        # Create bin edges for the grid cells
        x_edges = np.arange(x_min, x_max + delta_x, delta_x)
        y_edges = np.arange(y_min, y_max + delta_y, delta_y)
        
        width = len(x_edges) - 1
        height = len(y_edges) - 1
        counts = np.zeros((height, width), dtype=np.int32)

        # Each sounding represents a 20x20 m area; used to derive normalized density per cell
        point_area = 20 * 20
        points_per_cell = (delta_x * delta_y) / point_area
        
        for dl in dataloaders:
            print(f"Processing CDI ID: {dl.metadata['LOCAL_CDI_ID']}")
            easting = dl.data['Easting_N31'].to_numpy()
            northing = dl.data['Northing_N31'].to_numpy()
            
            # Use searchsorted to find bin indices for each point
            x_indices = np.searchsorted(x_edges, easting, side='right') - 1
            y_indices = np.searchsorted(y_edges, northing, side='right') - 1
            
            # Filter to valid cells (within grid bounds)
            valid_mask = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            
            # Use advanced indexing to increment cell counts (row=y, col=x)
            counts[y_indices, x_indices] += 1
        
        # Normalize to density (points per 20x20m equivalent) when requested
        raster_data = counts / points_per_cell if normalize else counts

        if plot:
            fig, ax = plt.subplots(figsize=(10, 8))
            extent = [x_min, x_max, y_min, y_max]
            max_val = raster_data.max()
            max_bin = int(np.ceil(max_val)) + 1  # include next integer bin edge
            boundaries = np.arange(-0.5, max_bin + 0.5, 1.0)
            cmap = plt.colormaps.get_cmap("tab20").resampled(len(boundaries) - 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N)

            img = ax.imshow(raster_data, extent=extent, origin='lower', cmap=cmap, norm=norm)
            label = 'Normalized Points (per cell)' if normalize else 'Point Count per cell'
            ax.set_title("Data Point Density Raster" if normalize else "Data Point Count Raster")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            ticks = np.arange(0, max_bin)
            fig.colorbar(img, ax=ax, label=label, orientation='horizontal', boundaries=boundaries, ticks=ticks)
            plt.show()
            
        # Save raster to file (normalized or raw based on flag)
        raster_path = "data_density_raster.tif" if normalize else "data_count_raster.tif"
        # Flip vertically: GeoTIFF convention is north at top (first row = max_y)
        raster_flipped = np.flipud(raster_data)
        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=raster_flipped.shape[0],
            width=raster_flipped.shape[1],
            count=1,
            dtype=raster_flipped.dtype,
            crs=self.crs,
            transform=rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, raster_flipped.shape[1], raster_flipped.shape[0]),
        ) as dst:
            dst.write(raster_flipped, 1)    
        
def hist():
    # open the tif and plot histogram
    with rasterio.open("data_count_raster.tif") as src:
        raster_data = src.read(1)
    plt.hist(raster_data.ravel(), bins=10)
    plt.title("Histogram of Data Density Raster")
    plt.xlabel("Density")
    plt.ylabel("Frequency")
    plt.show()
    
    # print statistics
    print("Raster Statistics:")
    print("Min:\t", np.min(raster_data))
    print("Max:\t", np.max(raster_data))
    print("Mean:\t", np.mean(raster_data))
    

if __name__ == "__main__":
    # Example usage
    aoi = AreaOfInterest("AOI.txt")
    # with open("data_loaders_v2.pkl", "rb") as f:
    #     dataloaders = pickle.load(f)
    #     print(f"Loaded {len(dataloaders)} data loaders from pickle.")
    
    # aoi.get_raster(dataloaders, delta_x=100, delta_y=100, plot=True, normalize=False)   
    
    # hist()
    
    #print lon lat of aoi transformed to epsg:4326
    aoi_wgs84 = aoi.aoi_gdf.to_crs(epsg=4326)
    for idx, row in aoi_wgs84.iterrows():
        print(f"AOI Polygon in WGS84 (EPSG:4326) - {idx}:")
        for coord in row['geometry'].exterior.coords:
            print(f"Lon: {coord[0]:.6f}, Lat: {coord[1]:.6f}")
    