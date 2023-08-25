import numpy as np
import timeit

from dda_atmos.steps import calc_threshold, calc_threshold_vectorized

a = np.random.rand(10000,1000)

args = {
    'data_mask':None,
    'downsample':0,
    'segment_length':5,
    'bias':60,
    'sensitivity':1,
    'quantile':90,
    'verbose':True
}

# check the thresholds calculated are the same
t1 = calc_threshold(a, **args)
t2 = calc_threshold_vectorized(a, **args)

print(f'non-equal elements: {np.sum(t1-t2 != 0)}')

print('timeit:')


args['verbose'] = False
print(timeit.timeit(lambda: calc_threshold(a, **args), number=15))
print(timeit.timeit(lambda: calc_threshold_vectorized(a, **args), number=15))

#non-equal elements: 9999
#timeit:
#23.493028390221298
#48.784387833904475
# unequal elements will be at either end of the matrix a, within segment_length of the ends of the first dimension.