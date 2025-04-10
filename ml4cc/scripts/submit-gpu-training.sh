#!/bin/bash

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o] [-c] [-p] [-m]
  -o : This is used to specify the output directory.
  -c : Train clusterization
  -p : Train peak finding
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
EOF
  exit 1
}

OUTPUT_EXISTS=false
RUN_ON_LUMI=true
TRAIN_CLUSTERIZATION=false
TRAIN_PEAK_FINDING=false
TRAIN_ONE_STEP=false
while getopts 'o:cpsm' OPTION; do
  case $OPTION in
    o)
        BASE_DIR=$OPTARG
        OUTPUT_EXISTS=true
        ;;
    c) TRAIN_CLUSTERIZATION=true ;;
    p) TRAIN_PEAK_FINDING=true ;;
    s) TRAIN_ONE_STEP=true ;;
    m) RUN_ON_LUMI=false ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"

echo Output will be saved into: $BASE_DIR
echo Training clusterization: $TRAIN_CLUSTERIZATION
echo Training peak finding: $TRAIN_PEAK_FINDING
echo Training one-step cluster counting: $TRAIN_ONE_STEP

if  [ "$RUN_ON_LUMI" = true ] ; then
    TRAINING_SCRIPT=ml4cc/scripts/submit-gpu-lumi.sh
else
    TRAINING_SCRIPT=ml4cc/scripts/submit-gpu-manivald.sh
fi

if [ "$TRAIN_CLUSTERIZATION" = true ] ; then
    sbatch $TRAINING_SCRIPT python3 ml4cc/scripts/train_clusterizer.py training.output_dir=$BASE_DIR
fi

if [ "$TRAIN_PEAK_FINDING" = true ] ; then
    sbatch $TRAINING_SCRIPT python3 ml4cc/scripts/train_peakFinder.py training.output_dir=$BASE_DIR
fi

if [ "$TRAIN_ONE_STEP" = true ] ; then
    sbatch $TRAINING_SCRIPT python3 ml4cc/scripts/train_one_step.py training.output_dir=$BASE_DIR
fi