import pickle

from difClass import DifferenceRaster, BathymetryRaster
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cmocean
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import time

def main():
    start = time.time()
    folder = "sandwave_detection_v1"
    while os.path.exists(folder):
        folder = folder[:-1] + str(int(folder[-1]) + 1)
    os.makedirs(folder)
    print(f"Saving results to folder: {folder}")
    
    raster_folder = "destriped_rasters"
    results = {}
    stop = 50
    i=0
    for filename in os.listdir(raster_folder):
        i += 1
        print(f"Processing file: {filename}, number of files left: {len(os.listdir(raster_folder)) - i}")
        os.mkdir(os.path.join(folder, filename.split('.')[0]))
        
        # if stop <= 0:
        #     break
        id = filename.split('.')[0]
        raster = np.load(os.path.join(raster_folder, filename))
        bat_raster = BathymetryRaster(raster)
        results[id] = {
            "bathymetry": bat_raster,
            "original": bat_raster.raster,
            "filled": bat_raster.raster_filled,
            "trend": bat_raster.trend,
            "residual": bat_raster.residual,
            "demeaned": bat_raster.raster_demeaned,
            "std": bat_raster.local_std(bat_raster.raster_filled, size=10),
            "grad": bat_raster.local_gradient(bat_raster.raster_filled, smooth=5),
            "grad_of_grad": bat_raster.local_gradient(bat_raster.local_gradient(bat_raster.raster_filled, smooth=5), smooth=5),
            "min": bat_raster.local_minmax(bat_raster.raster_filled, size=10)[0],
            "max": bat_raster.local_minmax(bat_raster.raster_filled, size=10)[1]
        }
        
        bat_raster.plot_raster(results[id]["original"], title=f"Original Raster - {id}", save_path=os.path.join(folder, id, f"{id}_original.png"))
        bat_raster.plot_raster(results[id]["residual"], title=f"Residual - {id}", save_path=os.path.join(folder, id, f"{id}_residual.png"))
        bat_raster.plot_raster(results[id]["std"], title=f"Local Std Dev - {id}", save_path=os.path.join(folder, id, f"{id}_std.png"))
        bat_raster.plot_raster(results[id]["grad"], title=f"Local Gradient - {id}", save_path=os.path.join(folder, id, f"{id}_grad.png"))
        bat_raster.plot_raster(results[id]["grad_of_grad"], title=f"Gradient of Gradient - {id}", save_path=os.path.join(folder, id, f"{id}_grad_of_grad.png"))
        bat_raster.plot_raster(results[id]["min"], title=f"Local Min - {id}", save_path=os.path.join(folder, id, f"{id}_min.png"))
        bat_raster.plot_raster(results[id]["max"], title=f"Local Max - {id}", save_path=os.path.join(folder, id, f"{id}_max.png"))
    
    pickle.dump(results, open(os.path.join(folder, "results_features.pkl"), "wb"))
    
    
def main2():
    folder = "sandwave_detection_v8"
    results = pickle.load(open(os.path.join(folder, "results_features.pkl"), "rb"))
    for id in results.keys():
        results[id]["feat"] = np.stack(
            [
                results[id]["residual"].flatten(),
                results[id]["std"].flatten(),
                results[id]["grad"].flatten(),
                results[id]["grad_of_grad"].flatten(),
                results[id]["min"].flatten(),
                results[id]["max"].flatten()
            ]
        )
    time1 = time.time()
    
    features = np.concatenate([results[key]["feat"] for key in results.keys()], axis=1).T
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    time2 = time.time()
    print(f"KMeans clustering completed in {time2 - time1:.2f} seconds.") 
    cmap = ListedColormap(['white', 'red', 'blue'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    idx = 0
    for key in results.keys():
        n_pixels = results[key]["feat"].shape[1]
        results[key]["labels"] = kmeans.labels_[idx:idx+n_pixels].reshape(results[key]["filled"].shape)
        results[key]["labels_smoothed"] = gaussian_filter(results[key]["labels"].astype(np.float32), sigma=40)
        results[key]["labels_smoothed"] = np.where(results[key]["labels_smoothed"] > 0.2, 1, 0)
        results[key]["labels"][results[key]["bathymetry"].nan_mask] = -1
        results[key]["labels_smoothed"][results[key]["bathymetry"].nan_mask] = -1
        idx += n_pixels
    
    
    
    for key in results.keys():
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(results[key]["original"], cmap=cmocean.cm.deep, origin='lower')
        plt.colorbar()
        plt.title(f"Original Raster \n {key}")
        plt.subplot(1, 3, 2)
        plt.imshow(results[key]["labels"], cmap=cmap, norm=norm, origin='lower')
        plt.colorbar()
        plt.title(f"KMeans Clusters \n {key}")
        plt.subplot(1, 3, 3)
        plt.imshow(results[key]["labels_smoothed"], cmap=cmap, norm=norm, origin='lower')
        plt.colorbar()
        plt.title(f"Smoothed KMeans Clusters \n {key}")
        plt.savefig(os.path.join(folder, f"{key}_clusters.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    if len(os.listdir(folder)) == 0:
        os.rmdir(folder)    
        

def main3():
    folder = os.listdir("destriped_rasters")
    name_set = set()
    for file in folder:
        name_set.add(file.split('_')[3])
    print(f"Unique CDI IDs in destriped rasters: {len(name_set)}")
        
if __name__ == "__main__":
    # main()
    main2()
    # main3()