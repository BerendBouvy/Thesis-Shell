import pickle
from difClass import DifferenceRaster, BathymetryRaster
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cmocean
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, label
import time
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import StandardScaler


def clean_smoothed_labels(
    labels,
    valid_mask,
    sigma=20,
    threshold=0.05,
    closing_iterations=2,
    min_cluster_pixels=200,
):
    """Smooth labels, bridge small gaps, and remove tiny connected components."""
    # Smooth only within valid pixels and normalize by local valid support so
    # edges are not artificially pulled toward zero.
    labels_f = labels.astype(np.float32)
    valid_f = valid_mask.astype(np.float32)
    weighted = gaussian_filter(labels_f * valid_f, sigma=sigma, mode="nearest")
    support = gaussian_filter(valid_f, sigma=sigma, mode="nearest")
    smoothed = np.divide(
        weighted,
        support,
        out=np.zeros_like(weighted, dtype=np.float32),
        where=support > 1e-6,
    )
    binary = smoothed > threshold
    binary &= valid_mask

    # Connect nearby fragments and fill small internal voids.
    # Keep edge-connected foreground from being eroded by implicit 0-padding at
    # array boundaries during closing.
    binary = binary_closing(
        binary,
        structure=np.ones((3, 3), dtype=bool),
        iterations=closing_iterations,
        border_value=1,
    )
    binary = binary_fill_holes(binary)

    # Remove tiny outlier islands to keep only larger segmentations.
    labeled_components, n_components = label(binary)
    if n_components > 0:
        component_sizes = np.bincount(labeled_components.ravel())
        keep = component_sizes >= min_cluster_pixels
        keep[0] = False
        binary = keep[labeled_components]

    output = np.zeros_like(labels, dtype=np.int8)
    output[binary] = 1
    output[~valid_mask] = -1
    return output

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
    if not os.path.exists(os.path.join(folder, "labels")):
        os.makedirs(os.path.join(folder, "labels"))
    results = pickle.load(open(os.path.join(folder, "results_features.pkl"), "rb"))
    ss = StandardScaler()
    for id in results.keys():
        results[id]["feat"] = np.stack(
            [
                np.abs(results[id]["residual"].flatten())**.5,
                results[id]["std"].flatten()**0.5,
                np.abs(results[id]["grad"].flatten())**.5,
                np.abs(results[id]["grad_of_grad"].flatten())**.5,
                # results[id]["min"].flatten()**2,
                # results[id]["max"].flatten()**2
            ]
        )
    time1 = time.time()
    
    features = np.concatenate([results[key]["feat"] for key in results.keys()], axis=1).T
    ss.fit(features)
    features = ss.transform(features)
    min_cluster_size = int(features.shape[0] * 0.05)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    print("Start clustering")
    kmeans = KMeansConstrained(n_clusters=2, size_min=min_cluster_size, random_state=0, verbose=True, n_init=1, tol=1e-3)
    kmeans.fit(features)
    time2 = time.time()
    # make sure that 0 is larger cluster and 1 is smaller cluster
    if np.sum(kmeans.labels_ == 0) < np.sum(kmeans.labels_ == 1):
        kmeans.labels_ = 1 - kmeans.labels_
    print(f"KMeans clustering completed in {time2 - time1:.2f} seconds.") 
    cmap = ListedColormap(['white', 'red', 'blue'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    idx = 0
    for key in results.keys():
        n_pixels = results[key]["feat"].shape[1]
        results[key]["labels"] = kmeans.labels_[idx:idx+n_pixels].reshape(results[key]["filled"].shape)
        valid_mask = ~results[key]["bathymetry"].nan_mask
        results[key]["labels_smoothed"] = clean_smoothed_labels(
            results[key]["labels"],
            valid_mask=valid_mask,
            sigma=20,
            threshold=0.05,
            closing_iterations=2,
            min_cluster_pixels=200,
        )
        results[key]["labels"][results[key]["bathymetry"].nan_mask] = -1
        idx += n_pixels
    
    
    
    for key in results.keys():
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(results[key]["original"], cmap=cmocean.cm.deep, origin='upper')
        plt.colorbar()
        plt.title(f"Original Raster \n {key}")
        plt.subplot(1, 3, 2)
        plt.imshow(results[key]["labels"], cmap=cmap, norm=norm, origin='upper')
        plt.colorbar()
        plt.title(f"KMeans Clusters \n {key}")
        plt.subplot(1, 3, 3)
        plt.imshow(results[key]["labels_smoothed"], cmap=cmap, norm=norm, origin='upper')
        plt.colorbar()
        plt.title(f"Smoothed KMeans Clusters \n {key}")
        plt.savefig(os.path.join(folder, f"{key}_clusters.png"), dpi=300, bbox_inches='tight')
        print(f"Saved cluster at {os.path.join(folder, f'{key}_clusters.png')}")
        plt.close()
        
        # save (smoothed) labels
        np.save(os.path.join(folder, "labels", f"{key}_labels.npy"), results[key]["labels"])
        np.save(os.path.join(folder, "labels", f"{key}_labels_smoothed.npy"), results[key]["labels_smoothed"])
        
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