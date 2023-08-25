'''Author: Andrew Martin
Creation Date: 24/8/23
'''

from dda_atmos.dda_data import DDA_data
from . import steps
from dda_data import DDA_data
import copy

class Batch:
    '''Class representing a batch of steps for executing in the DDA algorithm. A batch object can contain other batch objects to execute.'''

    def __init__(self):
        '''Initialisation function.'''
    
    def execute_batch(self, input_data: DDA_data):
        return copy.copy(input_data) # return a shallow copy of input_data, to allow a new instance that references data in the original, saving on memory...
    

class CreateKernal(Batch):
    def __init__(self,**kernal_args):
        '''Initialisation function. For kernal creation, requires the kernal parameters to be passed to it.
        
        INPUTS:


        '''
        super().__init__()
        self.kernal = steps.create_kernal.Gaussian(**kernal_args)

    def execute_batch(self, input_data: DDA_data):
        '''Sets the 'kernal' parameter on the new_data to the kernal.'''
        new_data = super().execute_batch(input_data)
        new_data.kernal = self.kernal
        return new_data
    

class CalculateDensity(Batch):
    def __init__(self, **density_parameters):
        super().__init__()
        '''Initialisation function. For density calculation, requires density calculation parameters to be passed.'''
        self.density_parameters = density_parameters

    def execute_batch(self, input_data: DDA_data, data_key: str = 'data_in'):
        new_data = super().execute_batch(input_data)
        '''Calculates the density field of the input_data[data_key] data.'''


class SinglePass(Batch):
    '''Batch of algorithm steps to create a density field, calculate the threshold and determine the cloud mask from that.'''

    def __init__(self, pass_num: int = 0):
        '''
        INPUTS:
            pass_num: int
                The order of the pass. This will append a number to output variables.
        '''
        super().__init__()
        self.pass_num = pass_num

    def execute_batch(self, input_data: DDA_data):
        new_data = super().execute_batch(input_data)
        '''SinglePass will run the batches for creating the kernal, density field, thresholds and cloudmask.'''