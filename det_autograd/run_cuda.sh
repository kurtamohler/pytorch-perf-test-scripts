#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

python $SCRIPTPATH/det_autograd_support.py cuda
