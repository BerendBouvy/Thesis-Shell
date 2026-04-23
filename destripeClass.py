import os
import numpy as np
import cmocean
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import Rbf, griddata, CloughTocher2DInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal as signal
import time

class Destriper:
    """
    Destriping class for bathymetric/hydrographic raster data.
    
    Removes stripe artifacts from survey data using FFT-based filtering.
    Supports polynomial or Gaussian detrending with configurable notch filtering.
    """
    
    def __init__(self, trend_param=3, style='line', width=5, pad_style='wrap', detrend='gaussian', save_plot=None):
        """
        Initialize Destriper with configuration parameters.
        
        Parameters:
        -----------
        trend_param : int, default=3
            Polynomial degree (polynomial) or Gaussian sigma (gaussian)
        style : str, default='line'
            Notch style: 'line' or 'cross'
        width : int, default=5
            Notch width in pixels
        pad_style : str, default='wrap'
            Padding mode for FFT: 'wrap', 'reflect', 'nearest', etc.
        detrend : str, default='gaussian'
            Detrending method: 'polynomial' or 'gaussian'
        save_plot : str, optional
            Directory to save plots. If None, plots are not saved to disk.
        """
        self.trend_param = trend_param
        self.style = style
        self.width = width
        self.pad_style = pad_style
        self.detrend = detrend
        self.save_plot = save_plot
        
        # Intermediate results
        self.nan_mask = None
        self.raster_original = None
        self.raster_filled = None
        self.fit = None
        self.residuals = None
        self.filtered_residuals = None
        self.angle_list = None
        self.angle = None
        self.destriped = None
        self.window = None
        self.notch = None
    
    def process(self, raster, plot=False, save_path=None):
        """
        Execute full destriping pipeline on input raster.
        
        Parameters:
        -----------
        raster : ndarray
            Input bathymetric raster (may contain NaNs)
        plot : bool, default=False
            Generate diagnostic plots
        save_path : str, optional
            Directory for diagnostic plots. When provided, plots are saved and not shown.
            
        Returns:
        --------
        destriped : ndarray
            Destriped raster with original NaN locations restored
        """
        if save_path is not None:
            self.save_plot = save_path
        if self.save_plot and not os.path.exists(self.save_plot):
            os.makedirs(self.save_plot)
        
        # Step 1: Store original and fill NaNs
        self.mean_value = np.nanmean(raster)
        self.raster_original = raster.copy()
        self.nan_mask = np.isnan(raster)
        raster_demeaned = raster - self.mean_value
        self.raster_filled_demeaned = self._fill_nans_main(raster_demeaned, sigma=25)
        self.raster_filled = self.raster_filled_demeaned + self.mean_value
        
        # Step 2: Detrend
        if self.detrend == 'polynomial':
            self._detrend_polynomial(plot=plot, save_path=self.save_plot)
        elif self.detrend == 'gaussian':
            self._detrend_gaussian(plot=plot, save_path=self.save_plot)
        else:
            raise ValueError(f"Unknown detrend method: {self.detrend}")
        
        # Step 3: FFT and angle detection
        self.windowed_residuals = self._apply_window(self.residuals, type='hann', param=0.1)
        self.padded_residuals_window, unpadded_slice = self._apply_padding(self.windowed_residuals)
        
        # step 3 without windowing
        self.padded_residuals_nowindow, _ = self._apply_padding(self.residuals)
        
        
        F_window = self._apply_fft(self.padded_residuals_window)
        F_nowindow = self._apply_fft(self.padded_residuals_nowindow, plot=plot, save_path=self.save_plot)
                
        self.angle, _, self.notch = self._find_angle(F_window, plot=plot, save_path=self.save_plot)
        
        # Step 4: Filter in frequency domain
        self.filter_frequency_domain(unpadded_slice, F_nowindow)
              
        # Step 5: Restore NaN mask and reconstruct
        if self.nan_mask is not None and np.any(self.nan_mask):
            self.filtered_residuals[self.nan_mask] = np.nan
        
        self.destriped = self.fit + self.filtered_residuals
        
        # Step 6: Plotting
        if plot:
            self._plot_results(save_path=self.save_plot)
        
        return self.destriped

    def filter_frequency_domain(self, unpadded_slice, F):
        """Apply notch filter in frequency domain."""
        self.F_filtered = F * (1 - self.notch)
        filtered_residuals = np.fft.ifft2(np.fft.ifftshift(self.F_filtered))
        imag_ratio = np.linalg.norm(np.imag(filtered_residuals)) / np.linalg.norm(np.real(filtered_residuals) + 1e-12)
        
        # print(f"imaginary to real ratio in filtered residuals: {imag_ratio:.2e}")
        
        filtered_residuals = np.real(filtered_residuals)
        self.filtered_residuals = self._unpad_data(filtered_residuals, unpadded_slice)
    
    def set_angle(self, angle):
        """Manually set stripe angle (for testing)."""
        self.angle_list = angle if isinstance(angle, list) else None
        self.angle = angle[0] if isinstance(angle, list) else angle
        # print(f"Manually set angle to: {self.angle}°")
            
    def _fill_nans_main(self, data, sigma=15, mode='nearest'):
        """Fill NaN values using Gaussian smoothing."""
        while np.isnan(data).any():
            # start = time.time()
            # data = self._extrapolate_to_square(data, method='polynomial', degree=2)
            data = self._extrapolate_to_square(data, method='nearest_smooth', degree=2, smooth=2)
            # data = self._fill_nans_griddata(data, method='cubic')
            # data = self._fill_nans_rbf(data, smooth=0.1)
            # data = self._fill_nans_0(data)
            # print("NaN values detected in raster. Applying smooth interpolation...")
            # data = self._fill_nans(data, sigma=sigma, mode=mode)
            # end = time.time()
            # print(f"NaN filling iteration completed in {end - start:.2f} seconds.")
        return data
    
    def _detrend_polynomial(self, plot=False, save_path=None):
        """Fit polynomial trend to data.

        Plot output is saved (not shown) when `save_path` is provided.
        """
        x = np.arange(self.raster_filled.shape[1])
        y = np.arange(self.raster_filled.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = self.raster_filled
        
        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
        
        poly = PolynomialFeatures(self.trend_param)
        XY_poly = poly.fit_transform(np.column_stack((X_flat, Y_flat)))
        
        model = LinearRegression()
        model.fit(XY_poly, Z_flat)
        
        Z_fit_flat = model.predict(XY_poly)
        self.fit = Z_fit_flat.reshape(Z.shape)
        self.residuals = Z - self.fit
        
        if plot:
            self._plot_detrend('polynomial', save_path=save_path)
    
    def _detrend_gaussian(self, plot=False, save_path=None):
        """Fit Gaussian trend to data.

        Plot output is saved (not shown) when `save_path` is provided.
        """
        self.fit = ndimage.gaussian_filter(self.raster_filled, sigma=self.trend_param)
        self.residuals = self.raster_filled - self.fit
        
        if plot:
            self._plot_detrend('gaussian', save_path=save_path)
    
    def _apply_padding(self, data):
        """Apply padding for FFT stability."""
        pad_width = max(data.shape) // 2
        data_padded = np.pad(data, pad_width, mode=self.pad_style)
        unpadded_slice = (
            slice(pad_width, pad_width + data.shape[0]),
            slice(pad_width, pad_width + data.shape[1])
        )
        return data_padded, unpadded_slice
    
    def _apply_fft(self, data, plot=False, save_path=None):
        """Compute 2D FFT and optionally plot diagnostics.

        When `save_path` is provided, diagnostic plots are saved and not shown.
        """
        F = np.fft.fft2(data)
        F_shifted = np.fft.fftshift(F)
        
        if plot:
            fig = plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Data")
            plt.imshow(data, origin='lower', cmap='viridis')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("FFT Magnitude")
            plt.imshow(np.log(np.abs(F_shifted) + 1), origin='lower', cmap='hot')
            plt.colorbar()
            plt.tight_layout()
            target_dir = save_path or self.save_plot
            if target_dir:
                plt.savefig(os.path.join(target_dir, "fft_magnitude.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return F_shifted
    
    def _create_notch(self, data, center_size=30):
        """Create notch filter mask."""
        notch = np.zeros_like(data)
        for i in range(-self.width, self.width+1):
            notch += np.eye(data.shape[0], data.shape[1], k=i)
            if self.style == 'cross':
                notch += np.eye(data.shape[0], data.shape[1], k=-i)
                
        # Remove central region to avoid filtering out main signal
        center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
        notch[center_y-center_size:center_y+center_size, center_x-center_size:center_x+center_size] = 0
        return notch
    
    def _find_angle(self, F, step=1, plot=False, save_path=None):
        """Detect dominant stripe angle in frequency domain.

        When `save_path` is provided, diagnostic plots are saved and not shown.
        """
        amplitude = np.log(np.abs(F) + 1)
        notch = self._create_notch(amplitude, center_size=15)
        angles = np.arange(0, 180, step)
        responses = []
                
        for theta in angles:
            rotated = ndimage.rotate(notch, angle=theta, reshape=False)
            response_theta = np.sum(rotated * amplitude)/np.sum(rotated)
            responses.append(response_theta)
        
        relative_height = []
        for i in range(len(responses)):
            neighbors_idx = np.roll(np.arange(len(responses)), -i+4)[:8]  # Get 4 neighbors on each side
            neighbors = np.array(responses)[neighbors_idx]
            relative_height.append(responses[i] - np.mean(neighbors))
            
        if self.angle is not None:
            angle = self.angle
            best_idx = np.argmin(np.abs(angles - angle))
            
        else:
            best_idx = np.argmax(relative_height)
            angle = angles[best_idx]
            
        best_notch = ndimage.rotate(notch, angle=angle, reshape=False)
        
        if len(self.angle_list) > 1:
            best_notch += ndimage.rotate(notch, angle=self.angle_list[1], reshape=False)
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            axes[0].set_title("Filtered F", pad=10)
            im0 = axes[0].imshow(amplitude*(1-best_notch), origin='lower', cmap='hot')
            fig.colorbar(im0, ax=axes[0], shrink=0.9)
            axes[1].set_title(f"Best Angle: {angle}° (Response: {relative_height[best_idx]:.2f})", pad=10)
            im1 = axes[1].imshow(amplitude, origin='lower', cmap='hot')
            fig.colorbar(im1, ax=axes[1], shrink=0.9)
            target_dir = save_path or self.save_plot
            if target_dir:
                plt.savefig(os.path.join(target_dir, "angle_detection.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            axes[0].set_title("Angle vs Response", pad=10)
            axes[0].plot(angles, responses, marker='o', markersize=3, linewidth=1.2)
            axes[0].scatter([angle], [responses[best_idx]], color='red', s=50, zorder=3)
            axes[0].set_xlabel("Angle (degrees)")
            axes[0].set_ylabel("Notch Response")
            axes[0].grid(alpha=0.3)

            axes[1].set_title("Relative Height vs Neighbors", pad=10)
            axes[1].plot(angles, relative_height, marker='o', markersize=3, linewidth=1.2)
            axes[1].scatter([angle], [relative_height[best_idx]], color='red', s=50, zorder=3)
            axes[1].set_xlabel("Angle (degrees)")
            axes[1].set_ylabel("Relative Height")
            axes[1].grid(alpha=0.3)
            
            if target_dir:
                plt.savefig(os.path.join(target_dir, "angle_response_diff.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return angle, relative_height[best_idx], best_notch
    
    def _unpad_data(self, data_padded, unpadded_slice):
        """Crop padding from data."""
        return data_padded[unpadded_slice]
    
    def _apply_window(self, data, type='hann', param=0):
        """Apply window function to data."""
        if type == 'hann':
            window_y = signal.windows.hann(data.shape[0])
            window_x = signal.windows.hann(data.shape[1])
            self.window = np.outer(window_y, window_x)
        elif type == 'tukey':
            window_y = signal.windows.tukey(data.shape[0], alpha=param)
            window_x = signal.windows.tukey(data.shape[1], alpha=param)
            self.window = np.outer(window_y, window_x)
        elif type == 'hamming':
            window_y = signal.windows.hamming(data.shape[0])
            window_x = signal.windows.hamming(data.shape[1])
            self.window = np.outer(window_y, window_x)
        else:
            raise ValueError(f"Unknown window type: {type}")
        eps = 1e-2
        self.window[self.window < eps] = eps  # Avoid division by zero
        return data * self.window
    
    def _unwindow(self, data):
        """Reverse windowing effect (if needed)."""
        
        if self.window is not None:
            if self.save_plot:
                fig = plt.figure(figsize=(6, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(self.padded_residuals_window, origin='lower')
                plt.title("Padded Residuals")
                
                plt.subplot(1, 3, 2)
                plt.imshow(data, origin='lower')
                plt.title("Data * Window")
                
                plt.subplot(1, 3, 3)
                plt.imshow(data / self.window, origin='lower')
                plt.title("Unwindowed Data")
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_plot, "window_debug.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Window debug plot saved to {self.save_plot}/window_debug.png")
            
            return data / self.window
        else:
            print("No window applied, skipping unwindowing step.")
            return data        
    
    def _plot_detrend(self, method, save_path=None):
        """Plot detrending results.

        If `save_path` is provided, the figure is saved and not shown.
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        
        im1 = axes[0].imshow(self.raster_filled, origin='lower', cmap=cmocean.cm.deep)
        axes[0].set_title("Original Data (Extrapolation Filled)", pad=10)
        fig.colorbar(im1, ax=axes[0], shrink=0.8)
        
        method_str = f"{method.capitalize()} Trend (degree={self.trend_param})" if method == 'polynomial' else f"Gaussian Trend (sigma={self.trend_param})"
        im2 = axes[1].imshow(self.fit, origin='lower', cmap=cmocean.cm.deep)
        axes[1].set_title(method_str, pad=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8)
        
        im3 = axes[2].imshow(self.residuals, origin='lower', cmap=cmocean.cm.deep)
        axes[2].set_title("Residuals", pad=10)
        fig.colorbar(im3, ax=axes[2], shrink=0.8)
        
        target_dir = save_path or self.save_plot
        if target_dir:
            plt.savefig(os.path.join(target_dir, f"{method}_detrend.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_results(self, save_path=None):
        """Plot final destriping results with consistent color scaling.

        If `save_path` is provided, figures are saved and not shown.
        """
        
        # Main destriping figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        
        im1 = axes[0].imshow(self.raster_original, origin='lower', cmap=cmocean.cm.deep)
        axes[0].set_title("Original Raster", pad=10)
        fig.colorbar(im1, ax=axes[0], shrink=0.8)
        
        im2 = axes[1].imshow(self.destriped, origin='lower', cmap=cmocean.cm.deep)
        axes[1].set_title("Destriped Raster", pad=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8)
        
        im3 = axes[2].imshow(self.raster_original - self.destriped, origin='lower', cmap=cmocean.cm.deep)
        axes[2].set_title("Difference (Original - Destriped)", pad=10)
        fig.colorbar(im3, ax=axes[2], shrink=0.8)
        
        target_dir = save_path or self.save_plot
        if target_dir:
            plt.savefig(os.path.join(target_dir, "destriped.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Residuals figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        
        im1 = axes[0].imshow(self.residuals, origin='lower', cmap=cmocean.cm.deep)
        axes[0].set_title("Residuals Before Filtering", pad=10)
        fig.colorbar(im1, ax=axes[0], shrink=0.8)
        
        im2 = axes[1].imshow(self.filtered_residuals, origin='lower', cmap=cmocean.cm.deep)
        axes[1].set_title("Residuals After Filtering", pad=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8)
        
        im3 = axes[2].imshow(self.residuals - self.filtered_residuals, origin='lower', cmap=cmocean.cm.deep)
        axes[2].set_title("Difference in Residuals", pad=10)
        fig.colorbar(im3, ax=axes[2], shrink=0.8)
        
        if target_dir:
            plt.savefig(os.path.join(target_dir, "residuals_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def _smooth_fill(data, sigma=40, mode='nearest'):
        """Fill NaN values using Gaussian smoothing."""
        mask = np.isnan(data)
        tmp = data.copy()
        tmp[mask] = 0
        
        smooth_data = ndimage.gaussian_filter(tmp, sigma=sigma, mode=mode)
        smooth_weight = ndimage.gaussian_filter((~mask).astype(float), sigma=sigma, mode=mode)
        
        filled = data.copy()
        safe_weight = np.where(smooth_weight == 0, np.nan, smooth_weight)
        filled[mask] = smooth_data[mask] / safe_weight[mask]
        return filled
    
    @staticmethod
    def _fill_nans(data, sigma=15, mode='nearest'):
        """Fill NaNs while preserving mean."""
        data_mean = np.nanmean(data)
        data_demeaned = data - data_mean
        array_filled = Destriper._smooth_fill(data_demeaned, sigma=sigma, mode=mode)
        array_filled += data_mean
        return array_filled
    
    @staticmethod
    def _fill_nans_0(data):
        """fill nans with zeros (not recommended)"""
        filled = data.copy()
        filled[np.isnan(filled)] = 0
        return filled
    
    @staticmethod
    def _fill_nans_rbf(raster, smooth=0.1):
        """
        Fill NaN regions using RBF (Radial Basis Function) interpolation.
        Smoothly extrapolates to fill entire square region.
        
        Parameters:
        -----------
        raster : ndarray
            2D array with NaN values to fill
        smooth : float, default=0.1
            RBF smoothness parameter (0 = exact fit, higher = smoother)
            
        Returns:
        --------
        filled : ndarray
            Raster with NaNs filled using smooth RBF extrapolation
        """
        h, w = raster.shape
        
        # Get valid data points
        valid_mask = ~np.isnan(raster)
        if not np.any(valid_mask):
            # All NaN, return zeros
            return np.zeros_like(raster)
        
        y_valid, x_valid = np.where(valid_mask)
        z_valid = raster[valid_mask]
        
        # Create RBF interpolant
        rbf = Rbf(x_valid, y_valid, z_valid, function='thin_plate', smooth=smooth)
        
        # Evaluate over entire grid
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        raster_filled = rbf(x_grid, y_grid)
        
        # Keep original valid data, fill NaNs
        raster_filled[valid_mask] = raster[valid_mask]
        
        return raster_filled
    
    @staticmethod
    def _fill_nans_griddata(raster, method='cubic'):
        """
        Fill NaN regions using griddata interpolation (fast alternative to RBF).
        
        Parameters:
        -----------
        raster : ndarray
            2D array with NaN values to fill
        method : str, default='cubic'
            Interpolation method: 'linear', 'cubic', or 'nearest'
            'cubic' provides smooth interpolation with good speed tradeoff
            
        Returns:
        --------
        filled : ndarray
            Raster with NaNs filled using griddata interpolation
        """
        h, w = raster.shape
        
        # Get valid data points
        valid_mask = ~np.isnan(raster)
        if not np.any(valid_mask):
            # All NaN, return zeros
            return np.zeros_like(raster)
        
        y_valid, x_valid = np.where(valid_mask)
        z_valid = raster[valid_mask]
        
        # Create regular grid
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Interpolate using griddata (much faster than RBF for large datasets)
        raster_filled = griddata(
            (x_valid, y_valid),
            z_valid,
            (x_grid, y_grid),
            method=method,
            fill_value=np.nanmean(z_valid)  # fallback for extrapolation gaps
        )
        
        # Keep original valid data, fill NaNs
        raster_filled[valid_mask] = raster[valid_mask]
        
        return raster_filled
    
    @staticmethod
    def _fill_nans_clough(raster):
        """
        Fill NaN regions using CloughTocher2D interpolation.
        Provides C1 smooth interpolation with much better speed than RBF.
        Best balance between quality and performance for bathymetric data.
        
        Parameters:
        -----------
        raster : ndarray
            2D array with NaN values to fill
            
        Returns:
        --------
        filled : ndarray
            Raster with NaNs filled using smooth CloughTocher2D interpolation
        """
        h, w = raster.shape
        
        # Get valid data points
        valid_mask = ~np.isnan(raster)
        if not np.any(valid_mask):
            # All NaN, return zeros
            return np.zeros_like(raster)
        
        y_valid, x_valid = np.where(valid_mask)
        z_valid = raster[valid_mask]
        
        # Build CloughTocher2D interpolator (C1 continuous, smooth)
        interp = CloughTocher2DInterpolator(
            np.column_stack([x_valid, y_valid]),
            z_valid,
            fill_value=np.nanmean(z_valid)  # fallback for extrapolation
        )
        
        # Evaluate over entire grid
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        raster_filled = interp(x_grid, y_grid)
        
        # Keep original valid data, fill NaNs
        raster_filled[valid_mask] = raster[valid_mask]
        
        return raster_filled
    
    @staticmethod
    def _extrapolate_to_square(raster, method='polynomial', degree=2, smooth=None):
        """
        Extrapolate data beyond convex hull to fill entire square smoothly.
        Unlike gap-filling methods, this extends the surface trend outward.
        
        Parameters:
        -----------
        raster : ndarray
            2D array with NaN regions to extrapolate into
        method : str, default='polynomial'
            'polynomial': fit global surface (recommended for smooth extrapolation)
            'nearest_smooth': extend nearest values + smooth (faster)
        degree : int, default=2
            Polynomial degree (1=planar, 2=quadratic, 3=cubic)
            
        Returns:
        --------
        filled : ndarray
            Raster with entire square filled by extrapolating data trends
        """
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