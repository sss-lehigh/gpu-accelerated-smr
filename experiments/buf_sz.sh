#!/bin/bash

# buf_sz.sh
#
# Experiment: End-to-End Performance vs. RDMA Buffer Size

OUTFILE="results/buf_sz.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,lat_avg,lat_50p,lat_99p,lat_99_9p,throughput' >"$OUTFILE"

# Fixed: 
# 
# Mat size: 512
# System size: n nodes
# Num state matrices: 5
# CPU enabled: true
# GPU enabled: true
# Serial: false
# DAG: true
BASE_ARGS="--mat-size 512 --num-state-mat 5 --cpu-enabled --gpu-enabled --mode DAG"
BUF_SIZES=(8 16 32 64 128 256 512)
for buf_size in "${BUF_SIZES[@]}"; do
	echo "Resetting..."
	reset-all
	reset-memcached
	EXTRA_ARGS="${BASE_ARGS} --buf-size $buf_size"
	echo "Launching experiment with buffer size ${buf_size}..."
	run_mu "$2"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done
