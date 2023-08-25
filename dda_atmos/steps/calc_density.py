'''Author: Andrew Martin
Creation Date: 16/6/23

Function to calculate the density field from data and a data mask.
'''

import numpy as np
import numba
from scipy.signal import convolve2d as scipy_convolve2d

def calc_density(data, data_mask, kernal, density_args, verbose=False):
    '''Function to calculate the density field from data and a data_mask using the provided kernal.
    
    INPUTS:
        data : np.ndarray
            nxm numpy array containing intensity data for n vertical profiles of m vertical height bins.

        data_mask : np.ndarray (dtype=boolean)
            nxm boolean numpy array containing a mask for valid data for the density calculation. 1s represent invalid data, 0s represent valid data.

        kernal : np.ndarray
            jxk numpy array (smaller than data) that defines the kernal the data is convolved by in the density field caluclation.

        density_args: dictionary
            dictionary containing additional arguments for the calculation of the density field, namely, the other arguments in scipy.signal.convolve2d.

    OUTPUTS:
        density : np.ndarray
            nxm numpy array containing the density field of data.
    '''
    print('==== dda.steps.calc_density()')
    density = convolve_masked(data, data_mask, kernal, **density_args)
    return density


def convolve_masked(data, mask, kernal, **kwargs):
    '''Function to perform a convolution of a kernal on masked data.
    
    The convention is that masked values are 1s in mask, and the data to convolve is 0s in the mask.
    For a convolution on masked data, this is essentially treating the nan values as 0, but also calculating the changed kernal normalisation due to the mask.

    !!!! The normalised density field is returned !!!!

    INPUTS:
        data : np.ndarray
            array containing the data for which we want to perform the masked convolution on

        mask : np.ndarray (dtype=boolean)
            array of the same shape as data that contains 1s where nans and masked values are, and 0s where the desired data exists.

        kernal : np.ndarray
            convolutional kernal

        kwargs : other arguments to be passed to np.convolve

    OUTPUTS:
        density : np.ndarray
            numpy array the same shape as data that contains the density field of data convolved with kernal.
    '''
    convargs = {'mode':'same', 'boundary':'symm'} # default convargs
    for arg in ['mode','boundary','fillvalue']:
        if arg in kwargs:
            convargs[arg] = kwargs[arg]

    norm = scipy_convolve2d(~mask, kernal, **convargs)
    masked_data = data.copy()
    masked_data[mask] = 0
    density = scipy_convolve2d(masked_data,kernal, **convargs)

    # normalise density field
    density[norm>0] = density[norm>0] / norm[norm>0]
    return density



def convolve2d(data, kernal, boundary='fill', fillvalue=0):
    '''Function to handle the convolution in 2d, with pre-processing of the input data to account for boundary conditions.
    
    INPUTS:
        data: np.ndarray
            (n,m)

        kernal: np.ndarray
            (nk,mk)

        boundary: str {'fill', 'wrap', 'symm'}, optional

        fillvalue: float, optional

    OUTPUTS:
        conv: np.ndarray
            (n,m)
    '''
    (n,m) = data.shape
    (nk,mk) = kernal.shape
    dn = int(nk//2+1)
    dm = int(mk//2+1)
    new_data = np.zeros((n+nk, m+mk), dtype=data.dtype)
    if boundary == 'fill':
        new_data[:] = fillvalue
    elif boundary == 'wrap':
        new_data[:dn, :dm] = data[n-dn:, m-dm:]
        new_data[:dn, dm:dm+m] = data[n-dn:, :]
        new_data[:dn, dm+m:] = data[n-dn:, :dm]

        new_data[n+dn:, :dm] = data[:dn, m-dm:]
        new_data[n+dn:, dm:dm+m] = data[:dn, :]
        new_data[n+dn:, dm+m:] = data[:dn, :dm]

        new_data[dn:dn+n, :dm] = data[:, m-dm:]
        new_data[dn:dn+n, dm+m:] = data[:, :dm]
    elif boundary == 'symm':
        new_data[:dn, :dm] = data[:dn:-1, :dm:-1]
        new_data[:dn, dm:dm+m] = data[:dn:-1, :]
        new_data[:dn, dm+m:] = data[:dn:-1, m-dm::-1]

        new_data[n+dn:, :dm] = data[n-dn::-1, :dm:-1]
        new_data[n+dn:, dm:dm+m] = data[n-dn::-1, :]
        new_data[n+dn:, dm+m:] = data[n-dn::-1, m-dm::-1]

        new_data[dn:dn+n, :dm] = data[:, :dm:-1]
        new_data[dn:dn+n, dm+m:] = data[:, m-dm::-1]

    new_data[dn: dn+n, dm:dm+m] = data
    conv = perform_conv(new_data, kernal)
    return conv


@numba.jit(nopython=True)
def perform_conv(data, kernal):
    '''Function to perform a 2d convolution of data with a kernal.
    
    It is expected that both the inputs will be 2d numpy ndarrays, and that data.size >= kernal.size on all dims.

    In this function, the convolution is only calculated for where the kernal and data fully overlap, so the returned numpy array will be smaller than data.
    
    INPUTS:
        data: np.ndarray
            (n,m) numpy array containing the pre-processed data for convolving

        kernal: np.ndarray
            (nk,mk) numpy array where nk<=n, mk<=m, defining the kernal for use in the convolution.

    OUTPUTS:
        conv: np.ndarray
            (n-nk, m-mk) numpy array containing the convolved data, where boundary effects aren't included. Boundaries can be allowed in the convolution via pre-processing data.
    '''
    new_kernal = np.flip(kernal)
    (n,m) = data.shape
    (nk,mk) = kernal.shape

    conv = np.zeros((n-nk, m-mk),dtype=data.dtype)

    for i, row in enumerate(conv):
        for j, cval in enumerate(row):
            for k, krow in enumerate(new_kernal):
                for l, kval in enumerate(krow):
                    conv[i,j] += data[i+k+1, j+l+1] * kval

    return conv