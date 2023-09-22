'''Author: Andrew Martin
Creation Date: 22/9/23

Script to test the speed of computation for the dask-ified implementation of the DDA against the pure numpy implementation.

Code is taken from the dask_vs_nondask.ipynb notebook
'''

print('DASK:')

import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

import os
import pathlib
import time

import dask

from dask.distributed import Client

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

########## DASK execution code

if __name__ == '__main__':
    print('starting client')
    client=Client(processes=False)

# 1. calc_density
# when used in map_overlap, should be as map_overlap( ch_calculate_density(kernal_n, density_args_n) , ...) to supply a valid function handle

def ch_calc_density(kernal,density_args):
    def tfunc(ch_data, ch_mask):
        return dda.steps.calc_density(ch_data, ch_mask, kernal, density_args)
    return tfunc

# 2. calc_threshold

def ch_calc_threshold(threshold_args):
    def tfunc(ch_density, ch_mask):
        return dda.steps.calc_threshold(ch_density, ch_mask, **threshold_args)
    return tfunc

# 3. calc_cloud_mask
# this one can be used as the handle in map_blocks(), rather than being explicitly called to return a handle.

def ch_calc_cloud_mask(ch_density, ch_threshold, ch_mask):
    return dda.steps.calc_cloud_mask(ch_density, ch_threshold, ch_mask, remove_small_clusters=0)

# 4. combine_masks
# needs to be called in the use of map_blocks to return a valid function handle. i.e. map_blocks( ch_combine_masks(), ... )

def ch_combine_masks(remove_small_clusters=300):
    def tfunc(*ch_masks):
        return dda.steps.combine_masks(ch_masks, remove_small_clusters=remove_small_clusters)
    return tfunc

# for use in map_overlap functions on the dask arrays

DEPTH1 = {0:kernal1.shape[0]//2, 1:0}
DEPTH2 = {0:kernal2.shape[0]//2, 1:0}

BOUNDARY = {0:'reflect', 1:'reflect'}

da_data = ds.backscatter_1.data
da_data_mask = np.isnan(da_data)

da_density1 = dask.array.overlap.map_overlap(ch_calc_density(kernal1, density_args), da_data, da_data_mask, depth=DEPTH1, boundary=BOUNDARY)
da_thresholds1 = dask.array.overlap.map_overlap(ch_calc_threshold(threshold_args), da_density1, da_data_mask, depth=DEPTH1, boundary=BOUNDARY, drop_axis=1, new_axis=1)
da_cloud_mask1 = dask.array.map_blocks(ch_calc_cloud_mask, da_density1, da_thresholds1, da_data_mask, dtype=bool)

da_data_mask = np.logical_or(da_data_mask, da_cloud_mask1)

da_density2 = dask.array.overlap.map_overlap(ch_calc_density(kernal2, density_args2), da_data, da_data_mask, depth=DEPTH2, boundary=BOUNDARY)
da_thresholds2 = dask.array.overlap.map_overlap(ch_calc_threshold(threshold_args2), da_density2, da_data_mask, depth=DEPTH2, boundary=BOUNDARY, drop_axis=1, new_axis=1)
da_cloud_mask2 = dask.array.map_blocks(ch_calc_cloud_mask, da_density2, da_thresholds2, da_data_mask, dtype=bool)

da_cloud_mask_combined = dask.array.map_blocks(ch_combine_masks(), da_cloud_mask1,da_cloud_mask2, dtype=bool)

da_cloud_mask_combined = da_cloud_mask_combined.persist()

#################################

time_end = time.time()

print(f'Dask computation time: {time_end - time_start}')
if __name__ == '__main__':
    print(client)