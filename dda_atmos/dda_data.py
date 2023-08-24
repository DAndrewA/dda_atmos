'''Author: Andrew Martin
Creation Date: 24/8/23
'''

import numpy as np

class DDA_data:
    '''Class for storing the data used in and output by the DDA algorithm.'''
    _REQ_VALS = ['data_in','heights','dem']


    def __init__(self, data_in: np.ndarray, 
                 heights: np.ndarray = None, 
                 dem: np.ndarray = None):
        '''Initialisation function.
        
        INPUTS:
            data_in: np.ndarray
                (n,m) numpy ndarray containing the data values to be used in the DDA-atmos algorithm. The data should consist of n vertical profiles, each with m vertical bins. Data should be normalised within adjacent profiles to be on a single intesnity scale (but the data need not be calibrated or de-noised).

            heights: np.ndarray (optional)
                (m,) numpy ndarray containing the height values at which the data is recorded.

            dem: np.ndarray (optional)
                (n,) numpy ndarray containing the heights of the digital elevaltion model, in the same units as the heights in heights.
        '''
        self.data_in = data_in
        self.heights = heights
        self.dem = dem

        # check the dimensions of data_in, heights and dem are valid
        if self.data_in.ndim != 2:
            raise ValueError(f'dimensions for data_in not correct. Should be 2, was {self.data_in.ndim}.')
        n,m = self.data_in.shape
        if self.heights is not None:
            if self.heights.ndim != 1:
                raise ValueError(f'dimsenions for heights not correct. Should be 1, was {self.heights.ndim}')
            h_m = self.heights.shape[0]
            if m != h_m:
                raise ValueError(f'shape of heights {self.heights.shape} incompatible with shape of data_in {self.data_in.shape}. Should be ({m},)')
        if self.dem is not None:
            if self.dem.ndim != 1:
                raise ValueError(f'dimsenions for dem not correct. Should be 1, was {self.dem.ndim}')
            d_n = self.dem.shape[0]
            if n != d_n:
                raise ValueError(f'shape of dem {self.dem.shape} incompatible with shape of data_in {self.data_in.shape}. Should be ({n},)')
            
        # list to store the outputs of batch processes as dictionaries of values
        self.batch_outputs = []
        

    def __getitem__(self, *args):
        '''Allows the user to index the data object as obj[key] or obj[batch,key]'''
        if len(args) == 1:
            key = args[0]
            if key == 'data_in':
                return self.data_in
            elif key == 'heights':
                return self.heights
            elif key == 'dem':
                return self.dem
            else:
                for out_dict in self.batch_outputs:
                    if key in out_dict:
                        return out_dict[key]
            raise KeyError(f'Key {key} not found in DDA_data object or its outputs.')
        elif len(args) == 2:
            batch = args[0]
            key = args[1]
            if key in self.batch_outputs[batch]:
                return self.batch_outputs[batch][key]
            else: raise KeyError(f'Key {key} not found in batch output {batch}.')
        raise ValueError('Key given as {args}, should be a string, or a length-2 list [int, string].')


    def __repr__(self):
        '''Creates a tabulated format for viewing the data structures contained in a DDA_data object.'''
        ret_str = 'INPUTS:\n'
        for key in DDA_data._REQ_VALS:
            ret_str += f'{key:<8} | '
            val = self[key]
            ret_str += f'{type(val).__name__:<16} | ' 
            ret_str += f'{val.shape} \n'

        ret_str += '\nOUTPUTS:\n'
        for i, out_dict in self.batch_outputs:
            ret_str += f'batch {i} output:\n'
            for k,v in out_dict.items():
                ret_str += f'{k:<20} | {type(v)}'
        
        return ret_str
