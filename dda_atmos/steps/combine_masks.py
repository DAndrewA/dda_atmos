'''Author: Andrew Martin
Creation Date: 16/6/23

A function to logically combine cloud masks, with the option to remove small clusters after the combination.
'''

import numpy as np
from skimage.morphology import remove_small_objects

def combine_masks(masks, remove_small_clusters=0, verbose=False):
    '''Function to logically combine multiple cloud masks, with the option to remove small clusters of pixels afterwards.
    
    INPUTS:
        masks : iterable of (n,m) np.ndarrays (dtype=boolean)
            Iterable of numpy ndarrays that are all boolean and the same shape. These are the individual cloud masks, with 1s indicating cloud and 0 indicating cloud-free pixels.

        remove_small_clusters : int
            Variable denoting the minimum cluster size for pixels to be considered in the cloud mask. If 0, no clusters of pixels will be removed.

        verbose : bool
            Flag for printing out debug statements
    
    OUTPUTS:
        combined_mask : np.ndarray
            nxm numpy array containing the logically combined cloud masks, that has had small clusters removed.
    '''
    if verbose: print('==== dda.steps.combine_masks')
    combined_mask = np.zeros_like(masks[0])
    for m in masks:
        combined_mask = np.logical_or(combined_mask, m)

    if remove_small_clusters > 0:
        if verbose: print('Removing small objects.')
        combined_mask = remove_small_objects(combined_mask, min_size=remove_small_clusters)
        if verbose: print('Small objects removed.')

    return combined_mask


def ch_combine_masks(remove_small_clusters=300):
    '''Function to create a function handle to combine cloud masks stored in chunked dask arrays
    
    INPUTS:
        remove_small_clusters: int
            Variable denoting the minimum cluster size for pixels to be considered in the cloud mask. If 0, no clusters of pixels will be removed.

    OUTPUTS:
        tfunc: function handle
            Function handle to be passed to dask.array.map_blocks
    '''
    def tfunc(*ch_masks):
        return combine_masks(ch_masks, remove_small_clusters=remove_small_clusters, verbose=False)
    return tfunc