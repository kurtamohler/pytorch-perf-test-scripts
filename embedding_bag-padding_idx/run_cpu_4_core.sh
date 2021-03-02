#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

OMP_NUM_THREADS=4 taskset 0xf0 python $SCRIPTPATH/embedding_bag-perf.py cpu
