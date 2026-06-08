import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
import time


folder_in = "destriped_rasters"
folder_out = "amplitude_rasters"
folder_differences = "difference_rasters"
folder_vars = "variance_rasters"

[os.makedirs(folder, exist_ok=True) for folder in [folder_out, folder_differences, folder_vars]]
metafile = "meta/metadata_with_density_flagged2.csv"
meta = pd.read_csv(metafile)


def main():
    rasters = load_raster_files()
    years = get_years_dict(rasters)
    for cell in rasters.keys():
        for id in rasters[cell].keys():
            raster = rasters[cell][id]
            year = years[cell][id]
            for id2 in rasters[cell].keys():
                year2 = years[cell][id2]
                if year2 > year:
                    raster2 = rasters[cell][id2]
                    nan_ratio = np.isnan(raster+raster2).sum() / np.size(raster)
                    if nan_ratio < 0.8:
                        difference = raster2 - raster
                        mean_diff = np.nanmean(difference)
                        demeaned_diff = difference - mean_diff
                        # max_min = np.nanmax(np.abs(demeaned_diff))
                        # title = f"Difference {year2} and {year} for Cell {cell}\nMean Difference: {mean_diff:.2f} m"
                        # save_path = os.path.join(folder_differences, f"cell_{cell}_diff_{year2}_{year}.png")
                        # plot_raster(demeaned_diff, cmap='bwr', vmin=-max_min, vmax=max_min,
                        #             title=title, xlabel="X pixel [20m]", ylabel="Y pixel [20m]", save_path=save_path)
                        # local_variance = local_var(demeaned_diff, size=50)
                        # title = f"Local Variance of Difference {year2} and {year} for Cell {cell}\nMean Difference: {mean_diff:.2f} m"
                        # save_path = os.path.join(folder_vars, f"cell_{cell}_local_var_diff_{year2}_{year}.png")
                        # plot_raster(local_variance, cmap=cmocean.cm.amp, vmin=0, vmax=np.nanpercentile(local_variance, 99),
                        #             title=title, xlabel="X pixel [20m]", ylabel="Y pixel [20m]", save_path=save_path)
                        # np.save(os.path.join(folder_vars, "Rasters", f"cell_{cell}_local_var_diff_{id2}_{id}.npy"), local_variance)
                        
                        dt = year2 - year
                        amplitude_norm = np.abs(demeaned_diff) / dt
                        if not "Rasters_amp" in os.listdir(folder_out):
                            os.makedirs(os.path.join(folder_out, "Rasters_amp"))
                        np.save(os.path.join(folder_out, "Rasters_amp", f"cell_{cell}_amplitude_{id2}_{id}.npy"), amplitude_norm)
                        title = f"Amplitude of Difference {year2} and {year} for Cell {cell}\nMean Amplitude: {np.nanmean(amplitude_norm):.2f} m/year"
                        save_path = os.path.join(folder_out, "figures", f"cell_{cell}_amplitude_{year2}_{year}.png")
                        if not "figures" in os.listdir(folder_out):
                            os.makedirs(os.path.join(folder_out, "figures"))
                        plot_raster(amplitude_norm, cmap=cmocean.cm.amp, vmin=0, vmax=np.nanpercentile(amplitude_norm, 99),
                                    title=title, xlabel="X pixel [20m]", ylabel="Y pixel [20m]", save_path=save_path)
                        
                    

def compute_difference_rasters(output_dir="difference_rasters/Rasters_diff"):
    """Compute all pairwise signed difference rasters from the destriped rasters.

    For each cell and every ordered pair of surveys (newer, older):
      - difference  = raster_newer - raster_older
      - demeaned    = difference - nanmean(difference)   (removes bulk offset)
      - normalised  = demeaned / dt                      (m/yr, signed)

    Files are written to `output_dir` as:
      cell_{cell}_diff_{id_newer}_{id_older}.npy

    Pairs with more than 80% NaN overlap are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)

    rasters = load_raster_files()
    years   = get_years_dict(rasters)

    n_saved = 0
    for cell in rasters:
        ids = list(rasters[cell].keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                year_a = years[cell][id_a]
                year_b = years[cell][id_b]
                if year_a is None or year_b is None:
                    continue

                # Ensure id_newer / id_older ordering
                if year_a >= year_b:
                    id_newer, id_older = id_a, id_b
                    year_newer, year_older = year_a, year_b
                else:
                    id_newer, id_older = id_b, id_a
                    year_newer, year_older = year_b, year_a

                dt = year_newer - year_older
                if dt == 0:
                    continue

                r_new = rasters[cell][id_newer]
                r_old = rasters[cell][id_older]

                nan_ratio = np.isnan(r_new + r_old).sum() / r_new.size
                if nan_ratio >= 0.8:
                    continue

                difference = r_new - r_old
                demeaned   = difference - np.nanmean(difference)
                normalised = demeaned / dt

                fname = f"cell_{cell}_diff_{id_newer}_{id_older}.npy"
                np.save(os.path.join(output_dir, fname), normalised)
                n_saved += 1

    print(f"Saved {n_saved} difference rasters to '{output_dir}'.")


def plot_raster(raster, cmap, vmin, vmax, title, xlabel, ylabel, save_path):
    plt.imshow(raster, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Difference in Bathymetry (m)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def get_years_dict(rasters):
    years = {}
    for cell in rasters:
        years[cell] = {}
        for id in rasters[cell].keys():
            years[cell][id] = get_year(id)
    return years
    
def load_raster_files():
    rasters = {}
    for filename in os.listdir(folder_in):
        cell, id = filename.split('_')[1], filename.split('_')[3]
        if cell not in rasters:
            rasters[cell] = {}
        arr = np.load(os.path.join(folder_in, filename), allow_pickle=True)
        rasters[cell][id] = arr.item() if arr.ndim == 0 else arr
    return rasters

def get_year(id):
    id = str(id).split('.')[0]  # Strip file extension if present
    row = meta[meta['CDI-record id'] == int(id)]
    if not row.empty:
        return int(str(row['Start Date'].values[0])[:4])  # Extract the year from the date string
    else:
        print(f"Warning: No metadata found for ID {id}")
        return None
    
def local_var(raster, size=10):
    from scipy.ndimage import generic_filter
    def nanstd_filter(values):
        return np.nanstd(values)**2
    return generic_filter(raster, nanstd_filter, size=size)
        
        
if __name__ == "__main__":
    start = time.time()
    # main()
    compute_difference_rasters()
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")