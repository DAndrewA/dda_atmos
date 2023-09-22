#!/bin/bash
python_script="$1"

conda activate jaspy_plus_s3fs

# execute the appropriate python script
pythontime $python_script