#!/bin/bash

# num_state_mat.sh
#
# Experiment: End-to-End Performance vs. Number of State Matrices

OUTFILE="results/num_state_mat.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,cons_lat_avg,e2e_lat_avg,goodput' >"$OUTFILE"

# Fixed:
#
# System size: n nodes
# Buf size: 1024
# Mat size: 512
# Serial: false
# DAG: true

# GPU Enabled: true 
# CPU Enabled: false 
BASE_ARGS="--buf-size 1024 --mat-size 512 --gpu-enabled --mode DAG"
NUM_STATE_MATS=(5 6 7 8 9 10)
echo "Resetting..."
reset-all
reset-memcached
for num_state_mat in "${NUM_STATE_MATS[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --num-state-mat $num_state_mat"
	echo "Launching experiment with ${num_state_mat} state matrices..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done

# GPU Enabled: false 
# CPU Enabled: true 
BASE_ARGS="--buf-size 1024 --mat-size 512 --cpu-enabled --mode DAG"
NUM_STATE_MATS=(5 6 7 8 9 10)
echo "Resetting..."
reset-all
reset-memcached
for num_state_mat in "${NUM_STATE_MATS[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --num-state-mat $num_state_mat"
	echo "Launching experiment with ${num_state_mat} state matrices..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done

# GPU Enabled: true 
# CPU Enabled: true 
BASE_ARGS="--buf-size 1024 --mat-size 512 --cpu-enabled --gpu-enabled --mode DAG"
NUM_STATE_MATS=(5 6 7 8 9 10)
echo "Resetting..."
reset-all
reset-memcached
for num_state_mat in "${NUM_STATE_MATS[@]}"; do
	EXTRA_ARGS="${BASE_ARGS} --num-state-mat $num_state_mat"
	echo "Launching experiment with ${num_state_mat} state matrices..."
	run_mu "${EXE_PATH}"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done
