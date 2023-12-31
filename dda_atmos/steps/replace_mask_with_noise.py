'''Author: Andrew Martin
Creation Date: 22/6/23

Script contaiing the function to replcae values in the data array that are in the cloud_mask with noise.
'''

import numpy as np

def replace_mask_with_noise(data, mask, mean, sd, vmin=0, seed=None, verbose=False):
    '''Function to replace the values denoted by cloud_mask in data with normally distributed noise.

    INPUTS:
        data : np.ndarray
            (n,m) numpy array containing the data field, which will be partially filled with noise.

        mask : np.ndarray (dtype=bool)
            (n,m) numpy array containing 1s where noise should be and 0s where the original data is to be kept.

        mean : np.ndarray
            (n,) numpy array of the noise spectrum means for each vertical profile.

        sd : np.ndarray
            (n,) numpy array containing the standard deviations of the noise spectra for each vertical profile.

        vmin : float
            Minimum value that the returned data will contain. This accounts for the fact that some noise values could fall below a physical threshold (i.e. 0).

        seed : None, int
            Value used to initialise the scipy.stats.rv_continuous.random_state object. None will result in randomness on each run, whereas a provided number will seed the random algorithm.

        verbose : bool
            Flag for printing debug statements to the output log.


    OUTPUTS:
        data : np.ndarray
            (n,m) numpy array, consisting of the data variable with some values replaced by noise.
    '''
    if verbose:
        print('==== dda.steps.replace_mask_with_noise()')
        print(f'{seed=}')

    noisy_data = data.copy()
    (n_prof, n_vert) = noisy_data.shape
    rs = (n_prof,1)

    noise = np.random.default_rng(seed).normal(loc=mean.reshape(rs), scale=sd.reshape(rs), size=noisy_data.shape)
    noise[noise < vmin] = vmin

    noisy_data[mask] = noise[mask]

    return noisy_data