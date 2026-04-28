#!/bin/bash

# num_state_mat.sh
#
# Experiment: End-to-End Performance vs. Number of State Matrices

OUTFILE="results/num_state_mat.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,cons_lat_avg,e2e_lat_avg,goodput' >"$OUTFILE"

# Fixed: 
# 
# System size: n nodes
# Buf size: 64
# Mat size: 512
# CPU enabled: true
# GPU enabled: true
# Serial: false
# DAG: true
BASE_ARGS="--buf-size 64 --mat-size 512 --cpu-enabled --gpu-enabled --mode DAG"
NUM_STATE_MATS=(1 2 3 4 5)
for num_state_mat in "${NUM_STATE_MATS[@]}"; do
	echo "Resetting..."
	reset-all
	reset-memcached
	EXTRA_ARGS="${BASE_ARGS} --num-state-mat $num_state_mat"
	echo "Launching experiment with ${num_state_mat} state matrices..."
	run_mu "$2"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done


