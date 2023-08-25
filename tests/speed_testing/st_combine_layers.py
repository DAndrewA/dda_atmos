import numpy as np
import timeit

from dda_atmos.steps import combine_layers_from_mask, combine_layers_from_mask_vectorized

a = np.random.rand(10000,1000).round()


cm1 = combine_layers_from_mask(a)
cm2 = combine_layers_from_mask_vectorized(a)

print(f'Non-equal elements: {np.sum(cm1 != cm2)}')

print(timeit.timeit(lambda: combine_layers_from_mask(a), number=15))
print(timeit.timeit(lambda: combine_layers_from_mask_vectorized(a), number=15))