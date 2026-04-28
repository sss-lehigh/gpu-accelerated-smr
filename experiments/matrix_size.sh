#!/bin/bash

# matrix_size.sh
#
# Experiment: End-to-End Performance vs. Matrix Dimension

OUTFILE="results/matrix_sz.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,cons_lat_avg,e2e_lat_avg,goodput' >"$OUTFILE"

# Fixed:
#
# System size: n nodes
# Buf size: 1024
# Num state matrices: 5
# Serial: false
# DAG: true

# GPU Enabled: true 
# CPU Enabled: false 
BASE_ARGS="--buf-size 1024 --num-state-mat 5 --gpu-enabled --mode DAG"
MAT_SIZES=(128 256 512 1024 2048)
echo "Resetting..."
reset-all
reset-memcached
for mat_size in "${MAT_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --mat-size $mat_size"
	echo "Launching experiment with matrix size ${mat_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done

# GPU Enabled: false 
# CPU Enabled: true 
BASE_ARGS="--buf-size 1024 --num-state-mat 5 --cpu-enabled --mode DAG"
MAT_SIZES=(128 256 512 1024 2048)
echo "Resetting..."
reset-all
reset-memcached
for mat_size in "${MAT_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --mat-size $mat_size"
	echo "Launching experiment with matrix size ${mat_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done


# GPU Enabled: true 
# CPU Enabled: true 
BASE_ARGS="--buf-size 1024 --num-state-mat 5 --cpu-enabled --gpu-enabled --mode DAG"
MAT_SIZES=(128 256 512 1024 2048)
echo "Resetting..."
reset-all
reset-memcached
for mat_size in "${MAT_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --mat-size $mat_size"
	echo "Launching experiment with matrix size ${mat_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done



