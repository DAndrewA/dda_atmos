# for testing the speed of the calc_density functions.
import numpy as np
from dda_atmos.steps import calc_density as cd
from dda_atmos.steps import create_kernal
import timeit


a = np.random.rand(10000,1000)
b = create_kernal.Gaussian(sigma_y=3, sigma_x=10, cutoff=1)

scipyconv = cd.scipy_convolve2d(a,b, mode='same')
myconv = cd.convolve2d(a,b)

print(f'Non-equal elements: {np.sum(scipyconv != myconv)}')

print('timeit comparison:')
print(timeit.timeit(lambda: cd.scipy_convolve2d(a,b, mode="same"), number=15))
print(timeit.timeit(lambda: cd.convolve2d(a,b), number=15))

#Non-equal elements: 8559931
#timeit comparison:
#44.35556841036305
#44.10100054414943