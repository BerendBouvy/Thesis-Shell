from matplotlib import pyplot as plt
import numpy as np

from difClass import DifferenceRaster, BathymetryRaster

def main():
    raster1 = np.load("destriped_rasters/cell_45_CDI_2174834_destriped.npy")
    raster2 = np.load("destriped_rasters/cell_45_CDI_2613170_destriped.npy")
    
    raster1 = np.load("destriped_rasters/cell_72_CDI_2174760_destriped.npy")
    raster2 = np.load("destriped_rasters/cell_72_CDI_3844672_destriped.npy")
    
    diff_raster = DifferenceRaster(raster1, raster2)
    diff_raster.analyse_spectrum()
    diff_raster.plot_raster(diff_raster.raster_demeaned, title="Difference Raster (Demeaned)", show=True)
    diff_raster.plot_raster(diff_raster.magnitude_spectrum, title="Magnitude Spectrum", show=True)
    # angles, responses = diff_raster.find_angle()
    # plt.figure(figsize=(10, 5))
    # plt.plot(angles, responses, marker='o')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Response')
    # plt.title('Response vs Angle')
    # plt.show()
    
    # diff_raster.double_cross((100, 100), (0, diff_raster.raster.shape[1]-1), plot=True)
    diff_raster.histogram(show=True)
    
    
def main2():
    raster = np.load("destriped_rasters/cell_72_CDI_2174760_destriped.npy")
    # raster = np.load("destriped_rasters/cell_10_CDI_3844666_destriped.npy")
    # raster = np.load("destriped_rasters/cell_30_CDI_3844669_destriped.npy")
    # raster = np.load("destriped_rasters/cell_77_CDI_3466117_destriped.npy")

    bat_raster = BathymetryRaster(raster)
    
    bat_raster.plot_raster(bat_raster.raster, title="Original Raster", show=True)
    bat_raster.plot_raster(bat_raster.raster_filled, title="Filled Raster", show=True)
    bat_raster.plot_raster(bat_raster.trend, title="Trend", show=True)
    bat_raster.plot_raster(bat_raster.residual, title="Residual", show=True)
    end = bat_raster.raster_demeaned.shape[1]-1
    
    # bat_raster.cross_section(
    #     bat_raster.residual, (end//2, 0), 
    #     (end//2, end), 
    #     plot=True
    # )
    
    labels, rasters = bat_raster.cluster_sandwaves(bat_raster.raster_filled)
    bat_raster.plot_raster(labels, title="KMeans Sandwave Clusters", show=True)
    
    for i in range(len(rasters)):
        bat_raster.plot_raster(rasters[i], title=f"Feature {i}", show=True)
    
    # # bat_raster.analyse_spectrum()
    # bat_raster.cross_section(
    #     bat_raster.residual, (100, 100), 
    #     (0, bat_raster.residual.shape[1]-1), 
    #     plot=True
    #     )

if __name__ == "__main__":
    # main()
    main2()
    input("Press Enter to exit...")