#!/bin/bash

#OMP_NUM_THREADS=1 numactl --membind 0 --cpubind 0 python $(dirname "$0")/norm_perf.py

OMP_NUM_THREADS=1 taskset 0x10 python $(dirname "$0")/norm_output_complex_perf.py
