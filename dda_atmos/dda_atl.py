'''Author: Andrew Martin
Creation Date: 20/6/23

Functions to run my implementation of the DDA-atmos algorithm on ATL09 data.
'''

import numpy as np
import xarray as xr

from . import steps

def dda_atl(data, heights, dem,
        kernal_args={}, density_args={}, threshold_args={}, 
        kernal_args2={}, density_args2={}, threshold_args2={},
        min_cluster_size=300, remove_clusters_in_pass=False, fill_clouds_with_noise=True, noise_altitude=10000,
        dem_tol=3, ground_width=3,
        verbose=False):
    '''Function to run the full DDA-atmos algorithm on the ATL09 data.
    
    This follows the threading described in the ATBD part 2 Section 21. 

    INPUTS:
        data : np.ndarray
            (n,m) numpy array, the ATL09 cab_prof data that will be the input for the DDA algorithm
    
        heights : np.ndarray
            (m,) numpy array that contains the heights of the vertical bins in the ATL09 data.

        dem : np.ndarray
            (n,) numpy array containing the heights of the reference Digital Elevation Model given in the ATL09 data.

        kernal_args, kernal_args2 : dict
            Dictionaries containing the additional parameters required for the kernal calculation, for passes 1 and 2.

        density_args, density_args2 : dict
            Dictionaries containing the additional parameters for the density calculation in passes 1 and 2.

        threshold_args, threshold_args2 : dict
            Dictionaries containing the additional parameters for the threshold calculation

        min_cluster_size : int
            Minimum connected-cluster size for pixels in the cloud mask to not be removed.

        remove_clusters_in_pass : bool
            Flag for removing small clusters in cloud_mask[1,2] if True. Otherwise, small clusters will only be removed after the masks have been combined.

        fill_clouds_with_noise : bool
            Flag to fill in the data within cloud_mask1 with noise before calculating the density2 field.

        noise_altitude : float
            Altitude above ground level for non-cloud data to be considered noisy.

        dem_tol : int
            The number of bins from the dem within which a "cloudy" pixel must fall in to be considered a ground return signal.

        ground_width : int
            The number of bins the ground signal is assumed to be spread over in the density field due to the kernal convolution.

        verbose : bool
            Flag for printing debug statements

    OUTPUTS: tuples
    1)
        layer_ : np.ndarray
            (n,n_layers) numpy arrays containing the derived layer top and bottom heights

    2)
        density_ : np.ndarray
            (n,m) numpy arrays containing the calculated density fields for each pass of the DDA.

    3)
        thresholds_ : np.ndarray
            (n,) numpy arrays containing the threshold values for cloudy pixels in each profile

    4)
        cloud_mask_ : np.ndarray (dtype=bool)
            (n,m) numpy arrays containing the masks for cloudy pixels (1s) and non-cloudy pixels (0s) from each pass of the dda.

    5) 
        ground_mask : np.ndarray (dtype=bool)
            (n,m) numpy array denoting values in the density field where the ground singla should be.

        ground_height : np.ndarray
            (n,) numpy array of heights either from the dem or from get_ground_bin denoting where the ground signal is considered to be.

        layer_mask_with_ground : np.ndarray (dtype=bool)
            (n,m) numpy array containing the consolidated layer_mask before the ground signal has been removed.

    6)
        cloud_mask_no_ground : np.ndarray (dtype=bool)
            (n,m) numpy array containing the mask for cloudy pixels, once the ground signal has been accounted for.

        layer_mask : np.ndarray (dtype=bool)
            (n,m) numpy array containing the consolidated layer mask used to determine layer_bot and layer_top.
    '''
    # Threading: 1: Run DDA-atmos to determination of combined decluster mask [section 3.1 to 3.5]
    data_mask = np.isnan(data)

    if verbose: print('******** Starting pass 1')

    kernal1 = steps.create_kernal.Gaussian(**kernal_args, verbose=verbose)
    density1 = steps.calc_density(data, data_mask, kernal1, density_args, verbose)
    thresholds1 = steps.calc_threshold(density1, data_mask, **threshold_args, verbose=verbose)
    if remove_clusters_in_pass:
        cloud_mask1 = steps.calc_cloud_mask(density1,thresholds1,data_mask, remove_small_clusters=min_cluster_size, verbose=verbose)
    else:
        cloud_mask1 = steps.calc_cloud_mask(density1,thresholds1,data_mask, verbose=verbose)

    if verbose: print('******** Starting pass 2')

    kernal2 = steps.create_kernal.Gaussian(**kernal_args2, verbose=verbose)
    # update the data_mask variable to include cloud_mask1.
    if fill_clouds_with_noise:
        # determine what the noise mean and sd are
        # data[cloud_mask1] = np.rand(data.size,mean,sd)[cloud_mask1]
        mean, sd = steps.calc_noise_at_height(data, cloud_mask1, heights, dem, noise_altitude, kernal_args['quantile'],verbose=verbose)
        data_with_noise = steps.replace_mask_with_noise(data,cloud_mask1,mean,sd,verbose=verbose)

        density2 = steps.calc_density_field(data_with_noise ,data_mask, kernal2, **density_args2, verbose=verbose)
        del data_with_noise
    else:
        data_mask = np.logical_or(data_mask, cloud_mask1)
        density2 = steps.calc_density(data, data_mask, kernal2, density_args2, verbose)
    thresholds2 = steps.calc_threshold(density2, data_mask, **threshold_args2, verbose=verbose)
    if remove_clusters_in_pass:
        cloud_mask2 = steps.calc_cloud_mask(density2,thresholds2,data_mask, remove_small_clusters=min_cluster_size, verbose=verbose)
    else:
        cloud_mask2 = steps.calc_cloud_mask(density2,thresholds2,data_mask, verbose=verbose)
    # create the combined cloud_mask variable
    cloud_mask_combined = steps.combine_masks((cloud_mask1,cloud_mask2), remove_small_clusters=min_cluster_size, verbose=verbose)

    if verbose: print('******** Finding ground signal')
    # determine within which signal bins the ground lies
    (ground_bin, ground_height) = steps.get_ground_bin(density1, cloud_mask_combined, heights, dem, dem_tol, verbose=verbose)

    layer_mask_with_ground = steps.combine_layers_from_mask_vectorized(cloud_mask_combined, verbose=verbose)

    if verbose: print('******** Removing ground from signal')
    # remove the ground bins from cloud_mask
    cloud_mask_no_ground, ground_mask = steps.remove_ground_from_mask(layer_mask_with_ground, ground_bin,cloud_mask_combined,ground_width,heights, verbose=verbose)

    # create a new layer_mask with the ground signal removed
    layer_mask = steps.combine_layers_from_mask_vectorized(cloud_mask_no_ground, verbose=verbose)

    if verbose: print('******** Calculating layer boundaries')
    num_cloud_layers, layer_bot, layer_top = steps.get_layer_boundaries(layer_mask,heights, verbose=verbose)

    return (num_cloud_layers, layer_bot, layer_top), (density1, density2), (thresholds1, thresholds2), (cloud_mask1, cloud_mask2), (ground_mask, ground_height, layer_mask_with_ground), (cloud_mask_no_ground, layer_mask)
