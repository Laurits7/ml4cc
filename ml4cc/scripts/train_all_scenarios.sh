#!/bin/bash
# This script trains all scenarios for the ML4CC project.


PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o] [-s] [-p] [-m] [-c]
  -o : This is used to specify the output directory.
  -s : Training scenario [Options: one_step, two_step, two_step_minimal, all]
  -d : Dataset to train on [Options: CEPC, FCC, all]
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
  -c : [Only if two_step training scenario] Clusterization model to be used in training [Options: CNN, DNN, RNN, (DGCNN), all]
EOF
  exit 1
}


TRAINING_SCENARIO=one_step
TRAINING_DATASET=FCC
RUN_ON_LUMI=true
TRAIN_TWO_STEP=false
TRAIN_TWO_STEP_MINIMAL=false
TRAIN_ONE_STEP=false
CLUSTERIZATION_MODEL=all
HOST=lumi
while getopts 'o:s:d:mc:' OPTION; do
  case $OPTION in
    o) BASE_DIR=$OPTARG ;;
    s) TRAINING_SCENARIO=$OPTARG ;;
    d) TRAINING_DATASET=$OPTARG ;;
    m)
        RUN_ON_LUMI=false
        HOST=manivald
        ;;
    c) CLUSTERIZATION_MODEL=$OPTARG ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"


declare -a DATASETS=("FCC" "CEPC")
declare -a 

if  [ "$RUN_ON_LUMI" = true ] ; then
    TRAINING_SCRIPT=ml4cc/scripts/submit-gpu-lumi.sh
else
    TRAINING_SCRIPT=ml4cc/scripts/submit-gpu-manivald.sh
fi


if [ "$TRAINING_DATASET" = "all" ] ; then
    echo Currently only single dataset is supported
fi


if [ "$TRAINING_SCENARIO" = "all" ] ; then
    echo Currently only single training scenario is supported
fi


sbatch $TRAINING_SCRIPT python3 ml4cc/scripts/train.py training.ouput_dir=$BASE_DIR dataset=$TRAINING_DATASET host=$HOST training.type=$TRAINING_SCENARIO
