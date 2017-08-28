#!/bin/bash

# run OpenSAT with hard coded models & configs found here and in /vagrant

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
BASEDIR=`dirname $SCRIPT`


filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"

export PATH=/home/${user}/anaconda/bin:$PATH

if [ $# -ne 1 ]; then
  echo "Usage: runOpenSAT.sh <audiofile>"
  exit 1;
fi

# let's get our bearings: set CWD to path of this script
cd $BASEDIR

# first features
SSSF/code/feature/extract-htk-vm.sh $1

# then confidences
/home/vagrant/anaconda/bin/python SSSF/code/predict/1-confidence-vm.py $BASEDIR/SSSF/data/feature/evl.med.htk/$basename.htk $basename


