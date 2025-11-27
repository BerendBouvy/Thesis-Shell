import pandas as pd
import os
from coordFunc import *
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime


data_folder = "data"
datums = ['World Geodetic System 84 (4326)',
       'World Geodetic System 84 / UTM zone 31N (32631)',
       'World Geodetic System 84 / UTM zone 32N (32632)']

# metadata = pd.read_csv("metadata copy.csv")
metadata = pd.read_csv("meta/metadata_with_density_flagged.csv")

class dataLoader:
    '''Class to load and process data files based on metadata.'''
    def __init__(self, metadata):        
        self.metadata = metadata
        self.id = metadata["CDI-record id"]
        self.data = self.load_data()
        self.get_coordinates()
        
        
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

    def plot_data(self, save_path=None, show=True, bbox=False):
        '''Plot the data points on a map with optional bounding box.'''
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            sc = ax.scatter(self.data['Lon'], self.data['Lat'], c=self.data['Mean (m)'], cmap='viridis', s=10)
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
        
        
def create_data_loaders():
    loaders = []
    for idx in tqdm(range(len(metadata))):
        if metadata.iloc[idx]['rejected'] == 0:
            sample_metadata = metadata.iloc[idx]
            loader = dataLoader(sample_metadata)
            loaders.append(loader)
        else:
            print(f"Skipping rejected dataset with CDI ID: {metadata.iloc[idx]['LOCAL_CDI_ID']}")
    print(f"Created {len(loaders)} data loaders.")

    with open("data_loaders.pkl", "wb") as f:
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
    
if __name__ == "__main__":
    # create_data_loaders()
    # create_plots()
    # test1()
    # check_col_names()
    # check_col_names2()
    # flag_weird_datasets()
    gantt_chart()