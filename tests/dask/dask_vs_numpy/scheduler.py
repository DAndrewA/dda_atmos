'''Author: Andrew Martin
Creation Date: 22/9/23

Schedules the dask_speed.py and numpy_speed.py scripts for execution, to allow dask speed  test on multiple cores thats not through the dask gateway.
'''

import os
import glob
import datetime as dt

script_np = '/home/users/eeasm/_scripts/dda_atmos/tests/dask/dask_vs_numpy/numpy_speed.py'
script_da = '/home/users/eeasm/_scripts/dda_atmos/tests/dask/dask_vs_numpy/dask_speed.py'

queue = 'par-single'
timemax = '00:15:00' # in hh:mm:ss format
outdir = '/home/users/eeasm/_scripts/dda_atmos/tests/dask/dask_vs_numpy'
memreq = '20G'
ncores = 1

##### NUNMPY
jobname = 'da_vs_np_np'
jobid = f'da_vs_np_np'

cmd = f'sbatch -p {queue} -t {timemax} -n {ncores} --mem={memreq} --job-name={jobname} -o {outdir}/{jobid}.out -e {outdir}/{jobid}.err runjob.sh {script_np}'

print(cmd)
os.system(cmd)



jobname = 'da_vs_np_da'
jobid = f'da_vs_np_da'

cmd = f'sbatch -p {queue} -t {timemax} -n {ncores} --mem={memreq} --job-name={jobname} -o {outdir}/{jobid}.out -e {outdir}/{jobid}.err runjob.sh {script_da}'

print(cmd)
os.system(cmd)


jobname = 'da_vs_np_da_4core'
jobid = f'da_vs_np_da_4core'

cmd = f'sbatch -p {queue} -t {timemax} -n 4 --mem={memreq} --job-name={jobname} -o {outdir}/{jobid}.out -e {outdir}/{jobid}.err runjob.sh {script_da}'

print(cmd)
os.system(cmd)