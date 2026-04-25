#!/bin/bash

# num_nodes.sh
#
# Experiment: End-to-End Performance vs. System Size

OUTFILE="results/num_nodes.csv"
echo 'system_size,mat_size,buf_size,num_state_mat,cpu_enabled,gpu_enabled,exe_mode,lat_avg,lat_50p,lat_99p,lat_99_9p,throughput' >"$OUTFILE"

# Fixed: 
# 
# Mat size: 512
# Buf size: 64
# Num state matrices: 5
# CPU enabled: true
# GPU enabled: true
# Serial: false
# DAG: true
EXTRA_ARGS="--mat-size 512 --buf-size 64 --num-state-mat 5 --cpu-enabled --gpu-enabled --mode DAG"

ORIG_MACHINES=("${MACHINES[@]}")
for i in $(seq 3 ${#ORIG_MACHINES[@]}); do
	MACHINES=("${ORIG_MACHINES[@]:0:$i}")
	echo "Resetting..."
	reset-all
	reset-memcached
	echo "Launching experiment with ${#MACHINES[@]} nodes..."
	run_mu "$2"
	grep -oP '\[PARSE\] \K.*' logs/log_0.txt >>"$OUTFILE"
done
