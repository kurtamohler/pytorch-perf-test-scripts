#!/bin/bash

#ATEN_CPU_CAPABILITY=default OMP_NUM_THREADS=1 taskset 0x10 python $(dirname "$0")/norm_perf.py

OMP_NUM_THREADS=1 taskset 0x10 python $(dirname "$0")/norm_perf.py
