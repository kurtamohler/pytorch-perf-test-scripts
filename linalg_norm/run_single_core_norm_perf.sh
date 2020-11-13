#!/bin/bash

OMP_NUM_THREADS=1 numactl --membind 0 --cpubind 0 python $(dirname "$0")/norm_perf.py
