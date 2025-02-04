#!/bin/bash

# keras is not used, but for some reason, it's imported somewhere and crashes if this is not specified
export KERAS_BACKEND=torch
apptainer exec -B /scratch/persistent,/local --env PYTHONPATH=`pwd`:`pwd`/ml4cc --nv /home/laurits/singularity/pytorch_w_geom.simg "$@"
