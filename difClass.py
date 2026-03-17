import numpy as np
from scipy import ndimage
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, generic_filter, sobel, minimum_filter, maximum_filter, distance_transform_edt
import cmocean
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

class RasterBase:
    """Base class for raster processing shared by bathymetry and difference rasters."""

    def __init__(self, raster, pixel_size=20):
        self.raster = raster
        self.pixel_size = pixel_size
        self.mean = np.nanmean(raster)
        self.raster_demeaned = raster - self.mean
        self.raster_fill()
        
        
    def raster_fill(self, zero_fill=False):
        # Fill NaN values with the mean of the raster for FFT processing
        self.nan_mask = np.isnan(self.raster_demeaned)
        if zero_fill:
            self.raster_filled = np.where(self.nan_mask, 0, self.raster_demeaned) 
        else:
            self.raster_filled = self.extrapolate_to_square(self.raster_demeaned, method='nearest_smooth', smooth=2)
    
    def raster_inv_fill(self, raster_filled):
        # Restore original NaN values after processing
        raster_filled[self.nan_mask] = np.nan
        return raster_filled
        
    def analyse_spectrum(self):
        # Window and pad the raster
        self.raster_padded, self.window = self.window_and_pad(self.raster_filled)
        
        # Perform 2D FFT
        self.fft = np.fft.fft2(self.raster_padded)
        self.fft_shifted = np.fft.fftshift(self.fft)
        
        # Compute magnitude spectrum
        self.magnitude_spectrum = np.log(np.abs(self.fft_shifted) + 1)  # Add 1 to avoid log(0)
        
    def find_angle(self):
        if not hasattr(self, "magnitude_spectrum"):
            self.analyse_spectrum()

        notch = np.eye(self.magnitude_spectrum.shape[0])
        angles = np.arange(0, 180, 1)
        responses = []
        
        for angle in angles:
            rotated_notch = ndimage.rotate(notch, angle, reshape=False)
            response = np.sum(self.magnitude_spectrum * rotated_notch)
            responses.append(response)
            
        return angles, responses
    
    def cross_section(self, raster, point1, point2, num=100, plot=False, save_path=None):
        """Extract a line profile between two points.

        If `plot` is True and `save_path` is provided, the plot is saved and not shown.
        """
        # Extract line profile between two points
        r1, c1 = point1
        r2, c2 = point2
        
        rows, cols = np.linspace(r1, r2, num=num), np.linspace(c1, c2, num=num)
        
        values = raster[rows.astype(int), cols.astype(int)]
        
        length = np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2) * self.pixel_size
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 2)
            plt.imshow(raster, cmap=cmocean.cm.deep)
            plt.plot([c1, c2], [r1, r2], 'r-', linewidth=2)
            plt.title("Raster with Cross-section Line")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.colorbar(label='Depth (m)')
            plt.subplot(1, 2, 1)
            plt.plot(np.linspace(0, length, num), values)
            plt.xlabel("Distance (m)")
            plt.ylabel("Depth (m)")
            plt.title("Cross-section")
            plt.grid(True)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        return values, length
    
    @staticmethod
    def extrapolate_to_square(raster, method='polynomial', degree=2, smooth=None):
        h, w = raster.shape
        valid_mask = ~np.isnan(raster)
        
        if not np.any(valid_mask):
            return np.zeros_like(raster)
        
        y_valid, x_valid = np.where(valid_mask)
        z_valid = raster[valid_mask]
        
        
        # Fit polynomial surface to valid data
        coords = np.column_stack([x_valid, y_valid])
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(coords)
            
        model = LinearRegression()
        model.fit(X_poly, z_valid)
            
        # Evaluate over entire grid (including outside convex hull)
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords_full = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        X_poly_full = poly.transform(coords_full)
        raster_filled = model.predict(X_poly_full).reshape(h, w)
            
        # Keep original valid data exact
        raster_filled[valid_mask] = raster[valid_mask]

        # Smooth the extrapolated regions more heavily
        smoothed = raster_filled.copy()
        if method == 'nearest_smooth':
            
            # Start with nearest neighbor
            indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
            raster_filled = raster[tuple(indices)]
            
            # Blend: keep data exact, smoothly transition in extrapolated region
            dist_from_data = distance_transform_edt(~valid_mask)
            blend_factor = np.clip(dist_from_data / 15, 0, 1)**2
            raster_filled = raster_filled * (1 - blend_factor) + smoothed * blend_factor
            raster_filled[valid_mask] = raster[valid_mask]
        
        
        for i in range(3):  
            if smooth is not None:
                raster_filled = ndimage.gaussian_filter(raster_filled, sigma=smooth)
                raster_filled[valid_mask] = raster[valid_mask]
                           
        return raster_filled
    
    @staticmethod
    def plot_raster(raster, title=None, show=False, save_path=None):
        """Plot a raster and optionally save it.

        When `save_path` is provided, the figure is saved and display is suppressed.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(raster, cmap=cmocean.cm.deep)
        plt.colorbar(label='Depth (m)')
        if title:
            plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show(block=False)
        else:
            plt.close()
                
        
        
    @staticmethod
    def window_and_pad(raster, pad_width=None):
        if pad_width is None:
            # Nearest power of 2 for padding
            size = 2**int(np.ceil(np.log2(raster.shape[0]))) - raster.shape[0]
            pad_width = [(size, size), (size, size)]
            
        # Apply window function
        window1d = signal.windows.hann(raster.shape[0])
        window2d = np.outer(window1d, window1d)
        
        raster_windowed = raster * window2d
        
        # Pad the raster
        raster_padded = np.pad(raster_windowed, pad_width, mode='constant', constant_values=0)
        
        return raster_padded, window2d


class BathymetryRaster(RasterBase):
    """Raster class for single bathymetry surfaces."""
    
    def __init__(self, raster, pixel_size=20):
        super().__init__(raster, pixel_size)
        self.detrend()
        
        # self.raster_filled = self.residual.copy()

    def detrend(self, sigma=30):
        """Remove large-scale trend and store residuals."""
        self.trend = gaussian_filter(self.raster_filled, sigma=sigma)
        self.residual = self.raster_filled - self.trend
        return self.residual
    
    def cluster_sandwaves(self, raster):
        """Use local features and KMeans to classify sandwave vs non-sandwave areas."""        
        std_raster = self.local_std(raster, size=10)
        grad_raster = self.local_gradient(raster, smooth=5)
        # grad_of_grad_raster = self.local_gradient(grad_raster, smooth=1)
        min_raster, max_raster = self.local_minmax(raster, size=15)
        
        # rasters = [std_raster, grad_raster, grad_of_grad_raster, min_raster, max_raster]
        rasters = [std_raster, grad_raster]
        
        features = np.stack([i.flatten() for i in rasters], axis=1)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        labels = kmeans.labels_.reshape(raster.shape)
        labels[self.nan_mask] = -1  # Mark NaN areas with a separate label
        return labels, rasters
        
    @staticmethod
    def local_std(raster, size=5):
        """Compute local standard deviation as a rough measure of roughness."""
        local_std_raster = generic_filter(raster, np.nanstd, size=size)
        return local_std_raster
    
    @staticmethod
    def local_gradient(raster, smooth=None):
        """Compute local gradient magnitude using Sobel filters."""
        gradient = np.sqrt(sobel(raster, axis=0)**2 + sobel(raster, axis=1)**2)
        if smooth is not None:
            gradient = gaussian_filter(gradient, sigma=smooth)
        return gradient

    @staticmethod
    def local_minmax(raster, size=15):
        """Compute local minima and maxima using minimum and maximum filters."""
        local_min = minimum_filter(raster, size=size)
        local_max = maximum_filter(raster, size=size)
        return local_min, local_max

class DifferenceRaster(RasterBase):
    """Raster class for differences between two bathymetry surfaces."""
    def __init__(self, raster, raster2, pixel_size=20):
        self.raster1 = raster
        self.raster2 = raster2
        self.raster = self.raster1 - self.raster2
        super().__init__(self.raster, pixel_size)
    

    def difference_stats(self):
        """Return bias/spread metrics for difference raster."""
        return {
            "bias": float(np.nanmean(self.raster)),
            "std": float(np.nanstd(self.raster)),
            "rmse": float(np.sqrt(np.nanmean(self.raster ** 2))),
            "p95_abs": float(np.nanpercentile(np.abs(self.raster), 95)),
        }
    
    def double_cross(self, point1, point2, num=100, plot=False, save_path=None):
        """Extract cross-sections from both rasters along the same line.

        If `plot` is True and `save_path` is provided, the figure is saved and not shown.
        """
        values1, length = self.cross_section(self.raster1, point1, point2, num=num)
        values2, _ = self.cross_section(self.raster2, point1, point2, num=num)
        
        if plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 2)
            plt.imshow(self.raster, cmap=cmocean.cm.deep, origin='upper')
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-', linewidth=2)
            plt.title("Difference Raster with Cross-section Line")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.colorbar(label='Depth Difference (m)')
            plt.subplot(1, 2, 1)
            plt.plot(np.linspace(0, length, num), values1, label='Raster 1')
            plt.plot(np.linspace(0, length, num), values2, label='Raster 2')
            plt.xlabel("Distance (m)")
            plt.ylabel("Depth (m)")
            plt.title("Cross-sections of Both Rasters")
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        return values1, values2
    
    def histogram(self, bins=50, title="Difference Histogram", show=False, save_path=None):
        """Plot a histogram of raster differences.

        When `save_path` is provided, the figure is saved and display is suppressed.
        """
        plt.figure(figsize=(6, 4))
        plt.hist(self.raster.flatten(), bins=bins, color='steelblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Depth Difference (m)")
        plt.ylabel("Frequency")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
        
        