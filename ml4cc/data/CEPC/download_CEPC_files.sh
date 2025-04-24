#!/bin/bash

# In order to get the file with all the download links, choose option "Get all URLs" at
# https://doi.org/10.57760/sciencedb.16322
DOWNLOAD_LINK_FILE="all_files.txt"

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: $PROGNAME [-s] [-d]
  -s : Use this flag to download the software
  -d : Use this flag to download the data
EOF
  exit 1
}

DOWNLOAD_SOFTWARE=false DOWNLOAD_DATA=false
while getopts 'sd' OPTION; do
  case $OPTION in
    s)
      DOWNLOAD_SOFTWARE=true
      ;;
    d)
      DOWNLOAD_DATA=true
      ;;
    ?)
      usage
      ;;
  esac
done
shift "$((OPTIND - 1))"

echo DOWNLOAD_DATA $DOWNLOAD_DATA
echo DOWNLOAD_SOFTWARE $DOWNLOAD_SOFTWARE

# Now download the chosen files.
# TODO: The structure of the raw data should be changed as such: 
# - Everything in the peakFinding and the train in clusterization goes to data/train
# - Everything in the test in clusterization goes to data/test
while read line; do
    URL=`echo $line | awk '{split($0,a,"&"); print a[1]}'`
    BROKEN_FILENAME=`echo $line | awk '{split($0,a,"&"); print a[2]}'`
    FILENAME_=`echo $BROKEN_FILENAME | awk '{split($0,a,"="); print a[2]}'`
    FILENAME="${FILENAME_:1}"
    if [[ $FILENAME =~ "samples" ]]; then
      if [[ $DOWNLOAD_DATA = true ]]; then
        FILENAME=${FILENAME//V2\/samples/data}
        FILENAME=${FILENAME//clusterization_test_samples/clusterization\/test}
        FILENAME=${FILENAME//clusterization_train_samples/clusterization\/train}
        FILENAME=${FILENAME//peakFinding_test_samples/peakFinding\/test}
        FILENAME=${FILENAME//peakFinding_train_samples/peakFinding\/train}  # TODO: Changes here
        curl $URL --create-dirs -o $FILENAME
      fi
    else
      if [[ $DOWNLOAD_SOFTWARE = true ]]; then
        curl $URL --create-dirs -o $FILENAME
      fi
    fi
done < $DOWNLOAD_LINK_FILE