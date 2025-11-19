import numpy as np
import pyproj as pp
import pandas as pd
import shapely.geometry as sg
import geopandas as gpd
import matplotlib.pyplot as plt

def split_dms(dms_str):
    [degrees, minutes, seconds] = dms_str.split('-')
    direction = seconds[-1]
    seconds = seconds[:-1]
    degrees, minutes, seconds = map(float, [degrees, minutes, seconds])
    return degrees, minutes, seconds, direction



def dms_to_dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def check_delimiter(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
        if '\t' in first_line:
            return '\t'
        else:
            return ' '
        
def convert_northing_easting(northing, easting, datum):
    if datum == 'World Geodetic System 84 / UTM zone 31N (32631)':
        transformer = pp.Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    elif datum == 'World Geodetic System 84 / UTM zone 32N (32632)':
        transformer = pp.Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    elif datum == 'World Geodetic System 84 (4326)':
        return northing, easting
    else:
        raise ValueError(f"Unsupported datum for conversion: {datum}")
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

def get_convex_hull(latitudes, longitudes, zone_number=31, plot=False):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(longitudes, latitudes), crs="EPSG:4326")
    
    utm_crs = pp.CRS(f"+proj=utm +zone={zone_number} +datum=WGS84 +units=m +no_defs")
    gdf_utm = gdf.to_crs(utm_crs.to_string())
    
    convex_hull_utm = gdf_utm.union_all().convex_hull

    area_sq_m = convex_hull_utm.area #m2
    
    resolution = len(gdf_utm) / area_sq_m # points per m2
    
    if plot:
        fig, ax = plt.subplots()
        fig.tight_layout()
        gdf.plot(ax=ax, color='blue', markersize=5, label='Data Points')
        # wrap the polygon into a GeoDataFrame with the UTM CRS, then reproject to WGS84 for plotting
        hull_gdf = gpd.GeoDataFrame(geometry=[convex_hull_utm], crs=gdf_utm.crs)
        hull_gdf = hull_gdf.to_crs("EPSG:4326")
        hull_gdf.boundary.plot(ax=ax, color='red', linewidth=2, label='Convex Hull')
        plt.legend()
        plt.show()
        
    return  area_sq_m*1e-6, resolution

def analyze_data_density(latitudes, longitudes, zone_number=31, plot=False):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(longitudes, latitudes), crs="EPSG:4326")
    
    utm_crs = pp.CRS(f"+proj=utm +zone={zone_number} +datum=WGS84 +units=m +no_defs")
    gdf_utm = gdf.to_crs(utm_crs.to_string())
    
    # create a grid to estimate local densities
    bounds = gdf_utm.total_bounds  # minx, miny, maxx, maxy
    grid_size = 100  # meters
    x_bins = np.arange(bounds[0], bounds[2] + grid_size, grid_size)
    y_bins = np.arange(bounds[1], bounds[3] + grid_size, grid_size)
    density, _, _ = np.histogram2d(gdf_utm.geometry.x, gdf_utm.geometry.y, bins=[x_bins, y_bins])
    
    ave_density = np.mean(density.flatten()[density.flatten()>0])
    # print("Density calculated over grid cells of size 100x100 meters.")
    print("Ave range density (points per grid cell):\t",  ave_density.round(3))
    if plot:
        plt.figure(figsize=(8, 6))
        plt.hist(density.flatten()[density.flatten()>0], bins=30, density=True)
        plt.xlabel('Points per 100x100 meter grid cell')
        plt.ylabel('Frequency')
        plt.title('Data Density Distribution')
        plt.show()
        
       
    return ave_density