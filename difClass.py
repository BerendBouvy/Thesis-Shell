import numpy as np
from scipy import ndimage
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, generic_filter, sobel, minimum_filter, maximum_filter, distance_transform_edt, rotate
import cmocean
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os

sandwave_detection_folder = os.path.join("sandwave_detection_v8", "labels")

class RasterBase:
    """Base class for raster processing shared by bathymetry and difference rasters."""

    def __init__(self, cell, cdi, pixel_size=20):
        self.cell = cell
        self.cdi = cdi
        self.raster = np.load(f"destriped_rasters/cell_{cell}_CDI_{cdi}_destriped.npy")
        self.pixel_size = pixel_size
        self.mean = np.nanmean(self.raster)
        self.raster_demeaned = self.raster - self.mean
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
        
    def find_angle(self, notch_angle=15):
        if not hasattr(self, "magnitude_spectrum"):
            self.analyse_spectrum()

        notch = self.cone_notch(self.magnitude_spectrum.shape[0], angle=notch_angle)
        angles = np.arange(0, 180, 5)
        responses = []
        
        
        for angle in angles:
            rotated_notch = ndimage.rotate(notch, angle, reshape=False)
            response = np.sum(self.magnitude_spectrum * rotated_notch)/np.sum(rotated_notch)
            responses.append(response)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(rotated_notch, cmap='gray')
            plt.title(f"Rotated Notch at {angle} degrees")
            plt.savefig(f"temp/notch_{angle}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        plt.figure(figsize=(10, 5))
        plt.plot(angles, responses, marker='o')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Response')
        plt.title('Response vs Angle')
        plt.grid(True)
        plt.show()
        
        best_angle = angles[np.argmax(responses)]
        best_angle_north = (180 - best_angle - 45)%360
        best_response = np.max(responses)
        
        self.best_angle = best_angle
        self.best_angle_north = best_angle_north
        
        best_notch = ndimage.rotate(notch, best_angle, reshape=False)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.magnitude_spectrum, cmap=cmocean.cm.deep)
        ax[0].set_title("Magnitude Spectrum")
        ax[0].set_xlabel("Frequency X")
        ax[0].set_ylabel("Frequency Y")
        ax[1].imshow(best_notch*self.magnitude_spectrum, cmap=cmocean.cm.deep)
        ax[1].set_title(f"Best Notch at {best_angle_north} degrees\nResponse: {best_response:.2f}")
        ax[1].set_xlabel("Frequency X")
        ax[1].set_ylabel("Frequency Y")
        plt.tight_layout()
        plt.show()   
                           
            
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
            shape = raster.shape
            pad_width = [(shape[0], shape[0]), (shape[1], shape[1])]
            
        # Apply window function
        window1d = signal.windows.hann(raster.shape[0])
        window2d = np.outer(window1d, window1d)
        
        raster_windowed = raster * window2d
        
        # Pad the raster
        raster_padded = np.pad(raster_windowed, pad_width, mode='constant', constant_values=0)
        
        return raster_padded, window2d

    @staticmethod
    def plot_spectrum(self, title="Magnitude Spectrum", show=False, save_path=None):
        if not hasattr(self, "magnitude_spectrum"):
            self.analyse_spectrum()
        self.plot_raster(self.magnitude_spectrum, title=title, show=show, save_path=save_path)

    @staticmethod
    def cone_notch(size, angle=15):
        trile = np.tril(np.ones((size, size)), k=0)
        trile_rotated1 = rotate(trile, angle, reshape=False, order=1, mode='reflect')
        trile_rotated2 = rotate(trile, -angle, reshape=False, order=1, mode='reflect')
        trile_combined = np.abs(trile_rotated2 - trile_rotated1)
        return trile_combined
        
    
class BathymetryRaster(RasterBase):
    """Raster class for single bathymetry surfaces."""
    
    def __init__(self, cell, cdi, pixel_size=20):
        super().__init__(cell=cell, cdi=cdi, pixel_size=pixel_size)
        self.detrend()
        
    def detrend(self, sigma=30):
        """Remove large-scale trend and store residuals."""
        self.trend = gaussian_filter(self.raster_filled, sigma=sigma)
        self.residual = self.raster_filled - self.trend
        return self.residual
    
    def get_cluster(self):
        path = os.path.join(sandwave_detection_folder, f"cell_{self.cell}_CDI_{self.cdi}_destriped_labels_smoothed.npy")
        self.cluster_labels = np.load(path)
        self.sw_ratio = np.nanmean(self.cluster_labels)
        return self.cluster_labels
    
    def set_angle(self, angle):
        self.angle = angle
    
    def find_best_cross(self):
        if not hasattr(self, "cluster_labels"):
            self.get_cluster()
        if not hasattr(self, "angle"):
            self.find_angle()
            self.set_angle(self.best_angle_north)
            
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

    
class DifferenceRaster(RasterBase):
    """Raster class for differences between two bathymetry surfaces."""
    def __init__(self, raster1, raster2, pixel_size=20):
        self.raster1 = raster1
        self.raster2 = raster2
        self.mean1 = np.nanmean(raster1)
        self.mean2 = np.nanmean(raster2)
        self.raster1_demeaned = self.raster1 - self.mean1
        self.raster2_demeaned = self.raster2 - self.mean2
        self.raster = self.raster1 - self.raster2
        self.raster_demeaned = self.raster1_demeaned - self.raster2_demeaned
        super().__init__(self.raster, pixel_size)
    

    def difference_stats(self):
        """Return bias/spread metrics for difference raster."""
        return {
            "bias": float(np.nanmean(self.raster)),
            "std": float(np.nanstd(self.raster)),
            "rmse": float(np.sqrt(np.nanmean(self.raster ** 2))),
            "p95_abs": float(np.nanpercentile(np.abs(self.raster), 95)),
        }
    
    def double_cross(self, point1, point2, num=100, show=False, save_path=None):
        """Extract cross-sections from both rasters along the same line.

        If `show` is True and `save_path` is provided, the figure is saved and not shown.
        """
        values1, length = self.cross_section(self.raster1, point1, point2, num=num)
        values2, _ = self.cross_section(self.raster2, point1, point2, num=num)
        
        if show:
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
        
    def minimise_shift(self, x_max=30, y_max=30, show=False, save_path=None):
        """Find optimal shift between rasters by minimizing difference variance."""
        best_shift = (0, 0)
        best_mean = np.inf
        shape = self.raster1_demeaned.shape
        best_diff=self.raster1_demeaned - self.raster2_demeaned
        scores = np.zeros((2*y_max+1, 2*x_max+1))
        for x_shift in range(-x_max, x_max + 1):
            for y_shift in range(-y_max, y_max + 1):
                shifted_raster1 = self.raster1_demeaned[max(0, y_shift):min(shape[0], shape[0]+y_shift),
                                                        max(0, x_shift):min(shape[1], shape[1]+x_shift)]
                shifted_raster2 = self.raster2_demeaned[max(0, -1*y_shift):min(shape[0], shape[0]+-1*y_shift), 
                                                        max(0, -1*x_shift):min(shape[1], shape[1]+-1*x_shift)]
                diff = shifted_raster1 - shifted_raster2
                mean = np.nanmean(diff)
                scores[y_shift+y_max, x_shift+x_max] = mean
                
                if mean < best_mean:
                    best_mean = mean
                    best_shift = (x_shift, y_shift)
                    best_diff = diff
        
        if show or save_path:
            fig, ax = plt.subplots(1, 4, figsize=(6, 6))
            ax[0].imshow(self.raster1_demeaned, cmap=cmocean.cm.deep)
            ax[0].set_title("Raster 1 Demeaned")
            #draw bbox of shifted area
            ax[0].add_patch(plt.Rectangle((max(0, best_shift[0]), max(0, best_shift[1])),
                                        min(shape[1], shape[1]+best_shift[0]), min(shape[0], shape[0]+best_shift[1]),
                                        fill=False, edgecolor='red', linewidth=2))
            ax[0].set_xlabel("X (pixels)")
            ax[0].set_ylabel("Y (pixels)")
            ax[1].imshow(self.raster2_demeaned, cmap=cmocean.cm.deep)
            ax[1].set_title("Raster 2 Demeaned")
            ax[1].add_patch(plt.Rectangle((max(0, -1*best_shift[0]), max(0, -1*best_shift[1])),
                                        min(shape[1], shape[1]+-1*best_shift[0]), min(shape[0], shape[0]+-1*best_shift[1]),
                                        fill=False, edgecolor='red', linewidth=2))
            ax[1].set_xlabel("X (pixels)")
            ax[1].set_ylabel("Y (pixels)")
            ax[2].imshow(best_diff, cmap=cmocean.cm.deep)
            ax[2].set_title("Best Difference")
            ax[2].set_xlabel("X (pixels)")
            ax[2].set_ylabel("Y (pixels)")
            im = ax[3].imshow(scores, cmap='viridis', extent=(-x_max, x_max, -y_max, y_max), origin='upper')
            ax[3].set_title("Mean Difference by Shift")
            ax[3].set_xlabel("X Shift (pixels)")
            ax[3].set_ylabel("Y Shift (pixels)")
            plt.colorbar(im, ax=ax[3], label='Mean Difference (m)')
            plt.suptitle(f"Best Shift: {best_shift} pixels, Mean Difference: {best_mean:.2f} m")
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            elif show:
                plt.show()
        return best_shift
        