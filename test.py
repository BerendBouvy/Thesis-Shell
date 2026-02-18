import pickle
from coordFunc import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dataloader import DataLoader


def test_conv_hull():
    latitudes = [52.0, 52.1, 52.2, 52.1]
    longitudes = [5.0, 5.1, 5.0, 4.9]
    area, res = get_convex_hull(latitudes, longitudes, zone_number=31, plot=True)
    # print("Convex Hull:", hull)
    print("Area (sq km):", area)
    print("Resolution (points per sq km):", res)
    
    
def test_conv_hull2():
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
    for loader in loaders:
        latitudes = loader.data['Lat'].to_list()
        longitudes = loader.data['Lon'].to_list()
        area, res = get_convex_hull(latitudes, longitudes, zone_number=31, plot=True)
        print(f"CDI ID: {loader.metadata['LOCAL_CDI_ID']} resolution: {res} points/sqm, area: {area} sqkm")
    
    
def test_density_analysis():
    metadata = pd.read_csv("metadata.csv")
    # add column 
    metadata['point_density(100x100m)'] = np.nan
    
    with open("data_loaders.pkl", "rb") as f:
        loaders = pickle.load(f)
    for loader in loaders:
        latitudes = loader.data['Lat'].to_list()
        longitudes = loader.data['Lon'].to_list()
        density = analyze_data_density(latitudes, longitudes, zone_number=31, plot=False)
        # print(f"CDI ID: {loader.metadata['LOCAL_CDI_ID']} density shape: {density.shape}")
        metadata.loc[metadata['LOCAL_CDI_ID'] == loader.metadata['LOCAL_CDI_ID'], 'point_density(100x100m)'] = density.round(3)
        
    metadata.to_csv("metadata_with_density.csv", index=False)
        
          
            
if __name__ == "__main__":
    # test_conv_hull()
    # test_conv_hull2()
    test_density_analysis()