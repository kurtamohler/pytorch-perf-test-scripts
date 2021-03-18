#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

OMP_NUM_THREADS=4 taskset 0xf0 python $SCRIPTPATH/clip_grad_norm-perf.py cpu
