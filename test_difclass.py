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
    
def main3():
    # raster1 = np.load("destriped_rasters/cell_72_CDI_2174760_destriped.npy")
    # raster2 = np.load("destriped_rasters/cell_72_CDI_3844672_destriped.npy")
    raster1 = np.load("destriped_rasters/cell_14_CDI_2612956_destriped.npy")
    raster2 = np.load("destriped_rasters/cell_14_CDI_3844669_destriped.npy")
    # raster1 = np.load("destriped_rasters/cell_58_CDI_2174813_destriped.npy")
    # raster2 = np.load("destriped_rasters/cell_58_CDI_2382040_destriped.npy")
       
    # diff_raster = DifferenceRaster(raster1, raster2)
    # diff_raster.plot_raster(diff_raster.raster, title="Difference Raster", show=True)   
    # diff_raster.minimise_shift(x_max=5, y_max=5, show=True)
    # shape = diff_raster.raster.shape
    # diff_raster.double_cross((shape[0]-1,0), (shape[0]//2, shape[1]//2), show=True)
    bath = BathymetryRaster(cell=14, cdi=2612956)
    # bath.find_angle(notch_angle=15)
    bath.get_cluster()
    plt.subplot(1, 2, 1)
    plt.imshow(bath.raster_filled, cmap='viridis')
    plt.subplot(1, 2, 2)
    plt.imshow(bath.cluster_labels, cmap='tab10')
    plt.show()
    
if __name__ == "__main__":
    # main()
    # main2()
    main3()
