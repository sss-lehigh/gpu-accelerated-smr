#!/bin/bash

# buf_sz.sh
#
# Experiment: End-to-End Performance vs. RDMA Buffer Size
OUTFILE="results/buf_sz.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,cons_lat_avg,e2e_lat_avg,goodput' >"$OUTFILE"
rm logs/* || true
# Fixed:
#
# Mat size: 512
# System size: n nodes
# Num state matrices: 5
# CPU enabled: true
# GPU enabled: true
# Serial: false
# DAG: true

# GPU Enabled: true 
# CPU Enabled: false 
BASE_ARGS="--mat-size 512 --num-state-mat 5 --gpu-enabled --mode DAG"
BUF_SIZES=(128 256 512 1024 2048 4096 8192 16384)
echo "Resetting..."
reset-all
reset-memcached
for buf_size in "${BUF_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --buf-size $buf_size"
	echo "Launching experiment with buffer size ${buf_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE" || true
done

# GPU Enabled: false 
# CPU Enabled: true 
BASE_ARGS="--mat-size 512 --num-state-mat 5 --cpu-enabled --mode DAG"
BUF_SIZES=(128 256 512 1024 2048 4096 8192 16384)
echo "Resetting..."
reset-all
reset-memcached
for buf_size in "${BUF_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --buf-size $buf_size"
	echo "Launching experiment with buffer size ${buf_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE" || true
done

# GPU Enabled: true 
# CPU Enabled: true 
BASE_ARGS="--mat-size 512 --num-state-mat 5 --cpu-enabled --gpu-enabled --mode DAG"
BUF_SIZES=(128 256 512 1024 2048 4096 8192 16384)
echo "Resetting..."
reset-all
reset-memcached
for buf_size in "${BUF_SIZES[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --buf-size $buf_size"
	echo "Launching experiment with buffer size ${buf_size}..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE" || true
done


