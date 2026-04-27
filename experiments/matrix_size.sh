#!/bin/bash

# matrix_size.sh
#
# Experiment: End-to-End Performance vs. Matrix Dimension

OUTFILE="results/matrix_sz.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,lat_avg,lat_50p,lat_99p,lat_99_9p,throughput' >"$OUTFILE"

# Fixed: 
# 
# System size: n nodes
# Buf size: 64
# Num state matrices: 5
# CPU enabled: true
# GPU enabled: true
# Serial: false
# DAG: true
BASE_ARGS="--buf-size 64 --num-state-mat 5 --cpu-enabled --gpu-enabled --mode DAG"
MAT_SIZES=(128 256 512 1024 2048)
for mat_size in "${MAT_SIZES[@]}"; do
	echo "Resetting..."
	reset-all
	reset-memcached
	EXTRA_ARGS="${BASE_ARGS} --mat-size $mat_size"
	echo "Launching experiment with matrix size ${mat_size}..."
	run_mu "$2"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done
