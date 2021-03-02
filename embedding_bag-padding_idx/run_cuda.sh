#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

python $SCRIPTPATH/embedding_bag-perf.py cuda
