import pickle

import pandas as pd
from dataloader import DataLoader
from matplotlib import pyplot as plt
import geopandas as gpd
import rasterio.features as rfeatures
import rasterio.transform as rtransform
import numpy as np
import cmocean

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import scipy.signal as sig
import scipy.ndimage as ndimage

import scipy.interpolate as interpolate

from dataloader import DataLoader

import os


def destripe_raster(raster, trend_param=3, plot=False, style='line', width=5, pad_style='wrap', detrend = 'gaussian', save_plot=None):
    if not os.path.exists(save_plot):
        os.makedirs(save_plot)
    nan_index = np.isnan(raster)
    while np.isnan(raster).any():
        print("NaN values detected in raster. Applying smooth interpolation to fill NaNs before destriping.")
        raster = fill_nans(raster, sigma=25)   
       
    # nan_index = None
    # if np.isnan(raster).any():
    #     print("NaN values detected in raster. Applying smooth interpolation to fill NaNs before destriping.")
    #     raster, nan_index = fill_nans(raster, sigma=15)
    #     if np.isnan(raster).any():
    #         print("Warning: NaN values still present after interpolation. Consider using a larger sigma or a different method to fill NaNs.")
    #         raster, nan_index2 = fill_nans_zero(raster)
    #         nan_index = nan_index | nan_index2
    if detrend == 'polynomial':
        fit, residuals = polynomial_fit(raster, degree=trend_param, plot=plot, save_plot=save_plot)
    elif detrend == 'gaussian':
        residuals, fit = detrend_gaussian(raster, sigma=trend_param, plot=plot, save_plot=save_plot)
    padded_residuals, unpadded_slice = apply_padding(residuals, style=pad_style)
    F = apply_fft(padded_residuals, plot=plot, save_plot=save_plot)
    notch = create_notch(padded_residuals, width=width, style=style)
    angle, _ = find_angle(F, notch, step=1, plot=plot, save_plot=save_plot)
    F_rotated = ndimage.rotate(F, angle=angle, reshape=False)
    F_filtered = F_rotated * (1 - notch)
    F_unrotated = ndimage.rotate(F_filtered, angle=-angle, reshape=False)
    filtered_residuals = np.fft.ifft2(np.fft.ifftshift(F_unrotated)).real
    filtered_residuals_unpadded = unpad_data(filtered_residuals, unpadded_slice)
    if nan_index is not None:
        print("Reapplying NaN mask to the destriped raster.")
        filtered_residuals_unpadded[nan_index] = np.nan
        raster[nan_index] = np.nan
    destriped = fit + filtered_residuals_unpadded
    
    if plot:
        bathy_stack = np.stack([raster, destriped])
        bathy_vmin = np.nanmin(bathy_stack)
        bathy_vmax = np.nanmax(bathy_stack)
        bathy_range = bathy_vmax - bathy_vmin
        residual_vmin = -bathy_range / 2
        residual_vmax = bathy_range / 2
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Raster")
        plt.imshow(raster, origin='lower', cmap=cmocean.cm.deep, vmin=bathy_vmin, vmax=bathy_vmax)
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title("Destriped Raster")
        plt.imshow(destriped, origin='lower', cmap=cmocean.cm.deep, vmin=bathy_vmin, vmax=bathy_vmax)
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("Difference (Original - Destriped)")
        plt.imshow(raster - destriped, origin='lower', cmap=cmocean.cm.deep, vmin=residual_vmin, vmax=residual_vmax)
        plt.colorbar()
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"destriped.png"), dpi=300, bbox_inches='tight')
        plt.close()
            
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Residuals Before Filtering")
        plt.imshow(residuals, origin='lower', cmap=cmocean.cm.deep, vmin=residual_vmin, vmax=residual_vmax)
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title("Residuals After Filtering")
        plt.imshow(filtered_residuals_unpadded, origin='lower', cmap=cmocean.cm.deep, vmin=residual_vmin, vmax=residual_vmax)
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("Difference in Residuals")
        plt.imshow(residuals - filtered_residuals_unpadded, origin='lower', cmap=cmocean.cm.deep, vmin=residual_vmin, vmax=residual_vmax)
        plt.colorbar()
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"residuals_comparison.png"), dpi=300, bbox_inches='tight')     
        plt.close()

    return destriped

def polynomial_fit(data, degree=3, plot=False, save_plot=None):
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = data

    # Flatten the arrays for polynomial fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Create polynomial features
    poly = PolynomialFeatures(degree)
    XY_poly = poly.fit_transform(np.column_stack((X_flat, Y_flat)))

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(XY_poly, Z_flat)

    # Predict the fitted values
    Z_fit_flat = model.predict(XY_poly)
    Z_fit = Z_fit_flat.reshape(Z.shape)
    residuals = Z - Z_fit
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Data")
        plt.imshow(Z, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.title(f"Polynomial Fit (degree={degree})")
        plt.imshow(Z_fit, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.title("Residuals")
        plt.imshow(residuals, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"polynomial_fit_degree_{degree}.png"), dpi=300, bbox_inches='tight')
        plt.close()
  

    return Z_fit, residuals

def detrend_gaussian(data, sigma=50, plot=False, save_plot=None):
    trend = ndimage.gaussian_filter(data, sigma=sigma)
    detrended = data - trend
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Data")
        plt.imshow(data, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.title(f"Gaussian Trend (sigma={sigma})")
        plt.imshow(trend, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.title("Detrended Data")
        plt.imshow(detrended, origin='lower', cmap=cmocean.cm.deep)
        plt.colorbar()
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"gaussian_detrend_sigma_{sigma}.png"), dpi=300, bbox_inches='tight')
        plt.close()
 
    return detrended, trend

def create_notch(data, width=5, style='cross'):
    notch = np.zeros_like(data)
    notch_y, notch_x = notch.shape
    notch[notch_y//2-width:notch_y//2+width, :] = 1
    if style == 'cross':
        notch[:, notch_x//2-width:notch_x//2+width] = 1
    notch[notch_y//2-width:notch_y//2+width, notch_x//2-width:notch_x//2+width] = 0
    return notch

def apply_fft(data, plot=False, save_plot=None):
    F = np.fft.fft2(data)
    F_shifted = np.fft.fftshift(F)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Data")
        plt.imshow(data, origin='lower', cmap='viridis')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title("FFT Magnitude")
        plt.imshow(np.log(np.abs(F_shifted) + 1), origin='lower', cmap='viridis')
        plt.colorbar()
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"fft_magnitude.png"), dpi=300, bbox_inches='tight')
        plt.close()

    return F_shifted

def apply_padding(data, style='wrap'):
    pad_width = max(data.shape) // 2
    data_padded = np.pad(data, pad_width, mode=style)
    # save index of original data in padded array for later cropping
    unpadded_slice = slice(pad_width, pad_width + data.shape[0]), slice(pad_width, pad_width + data.shape[1])
    return data_padded, unpadded_slice

def unpad_data(data_padded, unpadded_slice):
    return data_padded[unpadded_slice]

def find_angle(F, notch, step=1, plot=False, save_plot=None):
    amplitude = np.log(np.abs(F) + 1)
    angle = 0
    response = 0
    
    for theta in range(0, 90, step):
        rotated = ndimage.rotate(amplitude, angle=theta, reshape=False)
        response_theta = np.sum(rotated * notch)
        if response_theta > response:
            response = response_theta
            angle = theta
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Notch Filter")
        plt.imshow(notch, origin='lower', cmap='gray')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title(f"Best Angle: {angle}° (Response: {response:.2f})")
        rotated = ndimage.rotate(np.log(np.abs(F)+1), angle=angle, reshape=False)
        plt.imshow(rotated, origin='lower', cmap='viridis')
        plt.colorbar()
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(save_plot, f"angle_detection.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return angle, response


def smooth_fill(data, sigma=40, mode='nearest'):
    mask = np.isnan(data)
    
    # Replace NaNs with 0 for convolution
    tmp = data.copy()
    tmp[mask] = 0

    # Smooth the data and mask
    smooth_data = ndimage.gaussian_filter(tmp, sigma=sigma, mode=mode)
    smooth_weight = ndimage.gaussian_filter((~mask).astype(float), sigma=sigma, mode=mode)
    
    
    # Avoid divide by zero
    filled = data.copy()
    safe_weight = np.where(smooth_weight == 0, np.nan, smooth_weight)
    filled[mask] = smooth_data[mask] / safe_weight[mask]
    return filled

def fill_nans(data, sigma=15, mode='nearest'):
    data_mean = np.nanmean(data)
    data_demeaned = data - data_mean
    array_filled = smooth_fill(data_demeaned, sigma=sigma, mode=mode)
    array_filled += data_mean
    return array_filled

def fill_nans_zero(data):
    nan_index = np.isnan(data)
    data_filled = data.copy()
    data_filled[nan_index] = 0
    return data_filled


    
if __name__ == "__main__":
    metadata = pd.read_csv("meta/metadata_with_density_flagged2.csv")
    dl = DataLoader(metadata.iloc[53])
    # dl.plot_data(show=True)
    raster, _ = dl.get_raster((np.median(dl.data['Easting_N31']), np.median(dl.data['Northing_N31'])), 5000, 5000, cell_size=20)
    
    # # replace NaNs with median for testing
    # nan_index = np.isnan(raster)
    # array_filled = smooth_fill(raster, sigma=15)   
   

    os.makedirs("destriping_results", exist_ok=True)
    
    destriped = destripe_raster(raster, trend_param=3, plot=True, style='line', width=5, pad_style='wrap', detrend='gaussian', save_plot="destriping_results")