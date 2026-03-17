"""
Test script for destriping algorithm on a single dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
from destripeClass import Destriper
import cmocean

def main(save_path=None):
    """Run a single-dataset destriping test and generate comparison plots.

    If `save_path` is provided, all plots are saved in that directory.
    """
    # Load metadata
    metadata = pd.read_csv("meta/metadata_with_density_flagged2.csv")

    # Filter for useful datasets (not rejected, good density)
    useful_datasets = metadata[(metadata['rejected'] == 0) & (metadata['point_density(100x100m)'] > 20)]

    print(f"Found {len(useful_datasets)} useful datasets")

    # Select a dataset - let's use index 53 as shown in destriping.py example
    test_dataset = useful_datasets.iloc[47]  # You can change this index
    print(f"\nTesting with dataset: {test_dataset['LOCAL_CDI_ID']}")
    print(f"CDI-record id: {test_dataset['CDI-record id']}")
    print(f"Date range: {test_dataset['Start Date']} to {test_dataset['End Date']}")

    # Create dataloader
    dl = DataLoader(test_dataset)

    # Get a raster subset (5km x 5km with 20m cells)
    center_easting = np.median(dl.data['Easting_N31'])
    center_northing = np.median(dl.data['Northing_N31'])

    print(f"\nCreating raster centered at ({center_easting:.0f}, {center_northing:.0f})")
    raster, bbox = dl.get_raster(
        location=(center_easting, center_northing),
        width=5000,
        height=5000,
        cell_size=20,
        point_location='middle'
    )

    if raster is None:
        print("Failed to create raster - no data in region")
    else:
        print(f"Raster shape: {raster.shape}")
        print(f"Non-NaN pixels: {np.sum(~np.isnan(raster))}/{raster.size} ({100*np.sum(~np.isnan(raster))/raster.size:.1f}%)")
        print(f"Depth range: {np.nanmin(raster):.2f}m to {np.nanmax(raster):.2f}m")

        # Create destriper and process
        print("\n--- Starting destriping ---")
        output_dir = save_path or "test_destriping_output"
        destriper = Destriper(
            trend_param=3,      # Gaussian sigma for detrending
            style='line',        # Notch filter style
            width=3,             # Notch width
            pad_style='constant',    # Padding mode
            detrend='gaussian',   # Detrending method
            save_plot=output_dir
        )

        # Process with plotting enabled
        destriped = destriper.process(
            raster,
            plot=True
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(np.log(np.abs(destriper.F_filtered) + 1), cmap=cmocean.cm.deep, origin='lower')
        plt.colorbar()
        plt.title("Filtered FFT")
        plt.savefig(f"{output_dir}/filtered_fft.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n--- Destriping complete ---")
        print(f"Detected stripe angle: {destriper.angle}°")
        print(f"Destriped depth range: {np.nanmin(destriped):.2f}m to {np.nanmax(destriped):.2f}m")

        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        vmin = np.nanmin([raster, destriped])
        vmax = np.nanmax([raster, destriped])

        im1 = axes[0].imshow(raster, cmap=cmocean.cm.deep, vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Original Raster')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0], label='Depth (m)')

        im2 = axes[1].imshow(destriped, cmap=cmocean.cm.deep, vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title(f'Destriped (angle={destriper.angle}°)')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[1], label='Depth (m)')

        difference = raster - destriped
        diff_max = max(abs(np.nanmin(difference)), abs(np.nanmax(difference)))
        im3 = axes[2].imshow(difference, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, origin='lower')
        axes[2].set_title('Removed Stripes')
        axes[2].set_xlabel('X (pixels)')
        axes[2].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[2], label='Difference (m)')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"\nAll outputs saved to: {output_dir}/")


def main2(save_path=None):
    """Compare two NaN-extrapolation strategies visually.

    If `save_path` is provided, the figure is saved and not shown.
    """
    # raster = np.load("destriped_rasters/cell_72_CDI_2174760_destriped.npy")
    raster = np.load("destriped_rasters/cell_26_CDI_3844668_destriped.npy")
    destriper = Destriper()
    raster1 = destriper._extrapolate_to_square(raster, degree=2, method='polynomial')
    raster2 = destriper._extrapolate_to_square(raster, degree=2, method='nearest_smooth', smooth=2)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(raster1, cmap=cmocean.cm.deep, origin='lower')
    plt.colorbar()
    plt.title("Raster Extrapolated to Square (No Smoothing)")   
    plt.subplot(1, 2, 2)
    plt.imshow(raster2, cmap=cmocean.cm.deep, origin='lower')
    plt.colorbar()
    plt.title("Raster Extrapolated to Square (Smoothed)")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # main()
    main2()