#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

python $SCRIPTPATH/clip_grad_norm-perf.py cuda
