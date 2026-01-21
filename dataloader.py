import pandas as pd
import os
from shapely import Polygon
from coordFunc import *
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
from points import grid_heatmap
import rasterio.features as rfeatures
import rasterio.transform as rtransform
import pyproj as pp
import numpy as np



data_folder = "data"
datums = ['World Geodetic System 84 (4326)',
       'World Geodetic System 84 / UTM zone 31N (32631)',
       'World Geodetic System 84 / UTM zone 32N (32632)']

# metadata = pd.read_csv("metadata copy.csv")
metadata = pd.read_csv("meta/metadata_with_density_flagged2.csv")

class dataLoader:
    '''Class to load and process data files based on metadata.'''
    def __init__(self, metadata):        
        self.metadata = metadata
        self.id = metadata["CDI-record id"]
        print(f"Loading data for CDI ID: {self.metadata['LOCAL_CDI_ID']}")
        self.data = self.load_data()
        print(f"Loaded {self.data.shape[0]} data points.")
        print("Processing coordinates...")
        self.get_coordinates()
        print("Converting to UTM Zone 31N...")
        self.get_N31_coordinates()
        print("Saving Convex Hull...")
        self.convex_hull_utm, self.gdf_utm = self.get_convex_hull(zone_number=31, plot=False)
        print("Done.")
        
        
    def file_path(self):
        file_name = f"000574_XYZ_{self.metadata['LOCAL_CDI_ID'].replace('/', '_').replace('v', 'V')}.txt"
        return os.path.join(data_folder, file_name)

    def load_data(self):
        '''Load data from the specified file path.'''
        file_path = self.file_path()
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=0, delimiter=check_delimiter(file_path))
            if "Mean" in df.columns.to_list():
                df = df.rename(columns={"Mean": "Mean (m)"})
            return df
        else:
            print(f"File not found: {file_path}")
            return None
        
    def get_coordinates(self):
        '''Extract and convert coordinates to decimal degrees.'''
        if "Lat (DMS)" in self.data.columns and "Long (DMS)" in self.data.columns.to_list():
            self.data['Lat'] = self.data['Lat (DMS)'].apply(lambda x: dms_to_dd(*split_dms(x)))
            self.data['Lon'] = self.data['Long (DMS)'].apply(lambda x: dms_to_dd(*split_dms(x)))

        elif "Northing" in self.data.columns and "Easting" in self.data.columns.to_list():
            datum = self.metadata['Datum']
            lat, lon = convert_northing_easting(self.data['Northing'], self.data['Easting'], datum)
            self.data['Lat'] = lat
            self.data['Lon'] = lon
            
        else:
            print("No recognizable coordinate columns found.", self.data.columns.to_list())
    
    def get_N31_coordinates(self):
        if not 'Lat' in self.data.columns or not 'Lon' in self.data.columns:
            self.get_coordinates()
        transformer = pp.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
        self.data['Easting_N31'], self.data['Northing_N31'] = transformer.transform(self.data['Lon'].to_list(), self.data['Lat'].to_list()) 
        
        
    def plot_data(self, save_path=None, show=True, bbox=False):
        '''Plot the data points on a map with optional bounding box.'''
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            sc = ax.scatter(self.data['Lon'], self.data['Lat'], c=self.data['Mean (m)'], cmap='viridis', s=1)
            plt.colorbar(sc, label='Mean (m)')
            ax.coastlines()
            ax.set_title(f"Data Plot for {self.metadata['LOCAL_CDI_ID']}")
            if bbox:
                bb = self.get_bounding_box()
                scatter_points = {
                    'lon1': [bb[2], bb[2], bb[3], bb[3], bb[2]],
                    'lat1': [bb[0], bb[1], bb[1], bb[0], bb[0]]
                }
                ax.plot(scatter_points['lat1'], scatter_points['lon1'], color='red')
            if save_path:
                plt.savefig(save_path)
            if not show:
                plt.close()
            
        else:
            print("No data to plot.")
            
    def get_bounding_box(self):
        if self.data is not None:
            lat1 = self.metadata['Latitude 1']
            lat2 = self.metadata['Latitude 2']
            lon1 = self.metadata['Longitude 1']
            lon2 = self.metadata['Longitude 2']
            return (lat1, lat2, lon1, lon2)
        else:
            print("No data available.")
            return None
        
    def get_convex_hull(self, zone_number=31, plot=False):
        latitudes = self.data['Lat'].to_list()
        longitudes = self.data['Lon'].to_list()
        convex_hull_utm, gdf_utm = get_convex_hull(latitudes, longitudes, zone_number=zone_number, plot=plot)
        return convex_hull_utm, gdf_utm
        
    def __repr__(self):
        return f"dataLoader for CDI ID: {self.metadata['LOCAL_CDI_ID']} with {self.data.shape[0]} points."
    
    def __str__(self):
        return f"dataLoader for CDI ID: {self.metadata['LOCAL_CDI_ID']}"
    
    def __len__(self):
        return self.data.shape[0]
    
    def get_start_end_data(self):
        start = self.metadata["Start Date"] # yyyymmdd
        end = self.metadata["End Date"] # yyyymmdd
        # convert to datetime
        start = datetime.strptime(str(start), "%Y%m%d")
        end = datetime.strptime(str(end), "%Y%m%d")
        return start, end
    
    def get_raster(self, location, width, height, cell_size=20, point_location='middle'):
        if not 'Easting_N31' in self.data.columns or not 'Northing_N31' in self.data.columns:
            self.get_N31_coordinates()
            
        width = int(width/cell_size) * cell_size
        height = int(height/cell_size) * cell_size
        easting = self.data['Easting_N31'].to_numpy()
        northing = self.data['Northing_N31'].to_numpy()
        values = self.data['Mean (m)'].to_numpy()
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(easting, northing), crs="EPSG:32631")
        if point_location == 'middle':
            bbox = (location[0] - width/2, 
                    location[1] - height/2,
                    location[0] + width/2,
                    location[1] + height/2)
        elif point_location == 'lower_left':
            bbox = (location[0], 
                    location[1],
                    location[0] + width,
                    location[1] + height)
        bbox_polygon = Polygon([(bbox[0], bbox[1]),
                                (bbox[2], bbox[1]),
                                (bbox[2], bbox[3]),
                                (bbox[0], bbox[3])])
        points_in_bbox = points[points.within(bbox_polygon)]
        if points_in_bbox.empty:
            print("No points found in the specified bounding box.")
            return None, bbox
        values_in_bbox = values[points_in_bbox.index]
        raster = rfeatures.rasterize(
            ((geom, value) for geom, value in zip(points_in_bbox.geometry, values_in_bbox)),
            out_shape=(int(height/cell_size), int(width/cell_size)),
            transform=rtransform.from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], int(width/cell_size), int(height/cell_size)),
            fill=np.nan,
            all_touched=True,
            dtype='float32'
        )
        return raster, bbox
    
    def get_raster2(self, easting, northing, points_in_bbox, bbox, cell_size=20):
        values_in_bbox = self.data.loc[points_in_bbox.index, 'Mean (m)'].to_numpy()
        raster = rfeatures.rasterize(
            ((geom, value) for geom, value in zip(points_in_bbox.geometry, values_in_bbox)),
            out_shape=(int((bbox[3]-bbox[1])/cell_size), int((bbox[2]-bbox[0])/cell_size)),
            transform=rtransform.from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], int((bbox[2]-bbox[0])/cell_size), int((bbox[3]-bbox[1])/cell_size)),
            fill=np.nan,
            all_touched=True,
            dtype='float32'
        )
        return raster, bbox
    
def create_data_loaders():
    loaders = []
    for idx in tqdm(range(len(metadata))):
        if metadata.iloc[idx]['rejected'] == 0 and metadata.iloc[idx]['point_density(100x100m)'] > 20:
            sample_metadata = metadata.iloc[idx]
            loader = dataLoader(sample_metadata)
            loaders.append(loader)
        else:
            print(f"Skipping rejected ({metadata.iloc[idx]['rejected']}) or low-density ({metadata.iloc[idx]['point_density(100x100m)']}) dataset with CDI ID: {metadata.iloc[idx]['LOCAL_CDI_ID']}")
    print(f"Created {len(loaders)} data loaders.")

    with open("data_loaders_v2.pkl", "wb") as f:
        pickle.dump(loaders, f)
        
def create_plots():
    with open("data_loaders.pkl", "rb") as f:
        print("Loading data loaders...")
        loaders = pickle.load(f)
        print(f"Loaded {len(loaders)} data loaders.")
        
    if os.path.exists("plots") == False:
        os.mkdir("plots")
    
    num_plots_in_dir = len(os.listdir("plots"))
        
    for i, loader in tqdm(enumerate(loaders)):
        if i < num_plots_in_dir or loader.data.shape[1] < 5:
            
            pass
        else:
            name = loader.metadata['Data Set name']
            if "/" in name or "\\" in name:
                name = name.replace("/", "_").replace("\\", "_")
            plot_path = f"plots/{name}_plot_V1.png"
            while os.path.exists(plot_path):
                version = plot_path.split("_V")[-1].split(".png")[0]
                version_num = int(version) + 1
                plot_path = plot_path.replace(f"_V{version}.png", f"_V{version_num}.png")
                
            loader.plot_data(save_path=plot_path, show=False, bbox=True)

        
def test1():
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
    for loader in loaders:
        print(loader.get_bounding_box())
        print(loader.data.head())
        break
    
def check_col_names():
    for file in os.listdir(data_folder):
        # print first line of each file
        file_path = os.path.join(data_folder, file)
        with open(file_path, 'r') as f:
            first_line = f.readline()
            print(first_line)
            
def check_col_names2():
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
    for loader in loaders:
        if loader.data.columns.to_list()[2] != "Mean (m)":
            print(loader.metadata['Data Set name'])
            print(loader.data.columns.to_list())
            print(loader.data.head())
 
def flag_weird_datasets():
    df = pd.read_csv("metadata_with_density.csv")
    weird_ids = [2174832, 3455421] 
    
    df["rejected"] = df['CDI-record id'].apply(lambda x: 1 if int(x) in weird_ids else 0)
    df.to_csv("metadata_with_density_flagged.csv", index=False)
    
def gantt_chart():
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, loader in enumerate(loaders):
        if loader.metadata["point_density(100x100m)"] >20 and loader.metadata["rejected"] == 0:
            start, end = loader.get_start_end_data()
            ax.barh(i, (end - start).days, left=start, height=1)
            
        
    ax.set_xlabel('Date')
    ax.set_ylabel('Datasets')
    # ax.set_yticks(range(len(loaders)))
    # ax.set_yticklabels([loader.metadata['LOCAL_CDI_ID'] for loader in loaders])
    ax.invert_yaxis()
    plt.title('Gantt Chart of Dataset Collection Periods')
    plt.tight_layout()
    plt.savefig("plots/gantt_chart.png")
    plt.show()
    
def plot_number_of_points():
    # Open data loaders
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
        
    # Loop only through non-rejected datasets
    all_lats = []
    all_lons = []
    for loader in loaders:
        if loader.metadata["rejected"] == 0:
            all_lats.extend(loader.data['Lat'].to_list())
            all_lons.extend(loader.data['Lon'].to_list())
        
    #create a raster grid of 10000x10000m and count number of points in each cell
    grid_size = 10000  # meters
    utm_crs = pp.CRS(f"+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs")
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(all_lons, all_lats), crs="EPSG:4326")
    gdf_utm = gdf.to_crs(utm_crs.to_string())
    bounds = gdf_utm.total_bounds  # minx, miny, maxx, maxy
    # create a plot with the grid and colors based on number of points in each cell
    # use a method that can handle large number of points efficiently
    x_bins = np.arange(bounds[0], bounds[2] + grid_size, grid_size)
    y_bins = np.arange(bounds[1], bounds[3] + grid_size, grid_size)
    density, xedges, yedges = np.histogram2d(gdf_utm.geometry.x, gdf_utm.geometry.y, bins=[x_bins, y_bins])
    plt.figure(figsize=(10, 8))
    plt.imshow(density.T, origin='lower', cmap='hot', 
               extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    plt.colorbar(label='Number of Points per 1000x1000m cell')
    plt.title('Point Density Map (1000x1000m cells)')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.savefig("plots/point_density_1000m.png")
    plt.show()
    
        
def hm():
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
        
    all_lats = []
    all_lons = []
    for loader in loaders:
        if loader.metadata["rejected"] == 0:
            all_lats.extend(loader.data['Lat'].to_list())
            all_lons.extend(loader.data['Lon'].to_list())
            
    counts, (lat_edges, lon_edges) = grid_heatmap(
        latitudes=all_lats,
        longitudes=all_lons,
        cell_size=100,
        cmap='hot',
        show=True,
        save_path="plots/heatmap_all_data.png"
    )         
    
        
    
if __name__ == "__main__":
    # pass
    create_data_loaders()
    # create_plots()
    # test1()
    # check_col_names()
    # check_col_names2()
    # flag_weird_datasets()
    # gantt_chart()
    # plot_number_of_points()
    # hm()