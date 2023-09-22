'''Author: Andrew Martin
Creation Date: 22/9/23

Script to test the speed of computation for the dask-ified implementation of the DDA against the pure numpy implementation.

Code is taken from the dask_vs_nondask.ipynb notebook
'''

print('NUMPY:')

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

import os
import pathlib
import time

import dask

import dda_atmos as dda


STORE = pathlib.Path('/') / 'gws' / 'nopw' / 'j04' / 'icecaps' / 'ICECAPSarchive' / 'mpl' / 'zarrchive' / '202208.zarr'
CHUNK = {'time': 10000}

tslice = slice(dt.datetime(2022,8,10), dt.datetime(2022,8,20))

ds = xr.open_dataset(STORE, engine='zarr').sel(time=tslice).sel(height=slice(150,200000)).chunk(CHUNK)


kernal_args = {
    'sigma_y': 90, # pixel-wise if dy is unspecified
    'dy': 15, # in m. Thus, we expect 6 bins to account for 90m
    'sigma_x': 30, # estimated typical change time in seconds
    'dx': 5, # in seconds
    'cutoff': 1
}

density_args = {}

threshold_args = {
    'bias': 0.2,
    'sensitivity': 0.9,
    'quantile': 95
}

kernal_args2 = {
    'sigma_y': 90, # pixel-wise if dy is unspecified
    'dy': 15, # in m. Thus, we expect 6 bins to account for 90m
    'sigma_x': 60, # estimated typical change time in seconds
    'dx': 5, # in seconds
    'cutoff': 1
}

density_args2 = {}

threshold_args2 = {
    'bias': 0.2,
    'sensitivity': 0.8,
    'quantile': 90
}

kernal1 = dda.steps.create_kernal.Gaussian(**kernal_args)
kernal2 = dda.steps.create_kernal.Gaussian(**kernal_args2)

verbose = False

print(kernal1.shape, kernal2.shape)

time_start = time.time()

np_data = ds.backscatter_1.values
np_data_mask = np.isnan(np_data)

np_density1 = dda.steps.calc_density(np_data, np_data_mask, kernal1, density_args)
np_thresholds1 = dda.steps.calc_threshold(np_density1, np_data_mask, **threshold_args)
np_cloud_mask1 = dda.steps.calc_cloud_mask(np_density1, np_thresholds1, np_data_mask)

np_data_mask = np.logical_or(np_data_mask, np_cloud_mask1)

np_density2 = dda.steps.calc_density(np_data, np_data_mask, kernal2, density_args)
np_thresholds2 = dda.steps.calc_threshold(np_density2, np_data_mask, **threshold_args2)
np_cloud_mask2 = dda.steps.calc_cloud_mask(np_density2, np_thresholds2, np_data_mask)

cloud_mask_combined = dda.steps.combine_masks((np_cloud_mask1,np_cloud_mask2), remove_small_clusters=300)

time_end = time.time()

print(f'Numpy computation time: {time_end - time_start}')