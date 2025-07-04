#!/bin/bash
# This script trains all scenarios for the ML4CC project.

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o] [-s] [-d] [-e] [-m] [-c]
  -o : This is used to specify the output directory.
  -s : Training scenario [Options: one_step, two_step_pf, two_step_cl, two_step_minimal, all]
  -d : Dataset to train on [Options: CEPC, FCC, all]
  -e : Evaluation run [Default: False]
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
  -c : [Only if two_step training scenario] Clusterization model to be used in training [Options: CNN, DNN, RNN, (DGCNN), all]
EOF
  exit 1
}

EVALUATION=False
RUN_ON_LUMI=true
TRAIN_TWO_STEP=false
TRAIN_TWO_STEP_MINIMAL=false
TRAIN_ONE_STEP=false
CLUSTERIZATION_MODEL=DNN
HOST=lumi
while getopts 'o:s:d:emc:' OPTION; do
  case $OPTION in
    o) BASE_DIR=$OPTARG ;;
    s) TRAINING_SCENARIO=$OPTARG ;;
    d) TRAINING_DATASET=$OPTARG ;;
    e) EVALUATION=True ;;
    m)
        RUN_ON_LUMI=false
        HOST=manivald
        ;;
    c) CLUSTERIZATION_MODEL=$OPTARG ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"


# declare -a DATASETS=("FCC" "CEPC")
# declare -a SCENARIOS=("one_step" "two_step" "two_step_minimal")

echo Output will be saved into: $BASE_DIR
echo Training scenario: $TRAINING_SCENARIO
echo Training dataset: $TRAINING_DATASET
echo Running on: $HOST
echo Evaluation: $EVALUATION

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

# Apparently Hydra fails to set multilevel config in-place. This is just a workaround to load the correct config.
sed -i "/clusterization@clusterization.model/ s/: .*/: $CLUSTERIZATION_MODEL/" ml4cc/config/models/two_step/two_step.yaml

sbatch $TRAINING_SCRIPT python3 ml4cc/scripts/train.py training.output_dir=$BASE_DIR datasets@dataset=$TRAINING_DATASET environment@host=$HOST training.type=$TRAINING_SCENARIO training.model_evaluation=$EVALUATION
