#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

OMP_NUM_THREADS=1 taskset 0x10 python $SCRIPTPATH/embedding_bag-perf.py cpu
