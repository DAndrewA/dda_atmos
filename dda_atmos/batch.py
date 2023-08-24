'''Author: Andrew Martin
Creation Date: 24/8/23
'''

from dda_atmos.dda_data import DDA_data
from . import steps
from dda_data import DDA_data

class Batch:
    '''Class representing a batch of steps for executing in the DDA algorithm. A batch object can contain other batch objects to execute.'''

    def __init__(self):
        '''Initialisation function.'''
    
    def execute_batch(self, input_data: DDA_data):
        raise NotImplementedError('Use an instance of Batch that isnt the base Batch class.')
    

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
        return super().execute_batch(input_data)