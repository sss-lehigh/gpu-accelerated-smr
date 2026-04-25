#!/bin/bash

# mu.sh
#
# A script for uploading and running GPU-Accelerated SMR on CloudLab

set -e # Halt the script on any error

function make_screen {
	echo 'startup_message off' >>$1
	echo 'defscrollback 10000' >>$1
	echo 'autodetach on' >>$1
	echo 'escape ^jj' >>$1
	echo 'defflow off' >>$1
	echo 'hardstatus alwayslastline "%w"' >>$1
}

# Check the status of IBV on the target MACHINES
function check_ibv {
	echo "Checking ibv status:"
	for machine in ${MACHINES[@]}; do
		echo "$machine:"
		ssh $USER@$machine.$DOMAIN "ibv_devinfo -v"
	done
}

function cl_first_connect {
	echo "Performing one-time connection to CloudLab MACHINES, to get known_hosts right"
	for machine in ${MACHINES[@]}; do
		ssh -o StrictHostKeyChecking=no $USER@$machine.$DOMAIN echo "Connected"
	done
}

function install_deps {
	config_command=prepare_to_run.sh          # The script to put on remote nodes
	last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes

	# Names of packages that we need to install on CloudLab
	package_deps="librdmacm-dev ibverbs-utils libnuma-dev gdb libgtest-dev libibverbs-dev libmemcached-dev memcached libevent-dev libhugetlbfs-dev numactl libgflags-dev libssl-dev"
	# First-time SSH
	cl_first_connect

	# Build a script to run on all the MACHINES
	tmp_script_file="$(mktemp)" || exit 1
	echo 'echo `hostname`' >${tmp_script_file}
	# Turn off interactive prompts
	echo "sudo sed -i 's|http://us.archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources" >>${tmp_script_file}
	echo 'sudo apt update' >>"${tmp_script_file}"
	echo "sudo apt upgrade -y" >>${tmp_script_file}
	echo "sudo apt install -y ${package_deps}" >>${tmp_script_file}
	echo "echo 'kernel.perf_event_paranoid=-1' | sudo tee -a /etc/sysctl.conf" >>${tmp_script_file}
	echo "sudo sysctl -p" >>${tmp_script_file}

	# --- CUDA toolkit install (skipped on non-GPU nodes) ---
	echo 'if lspci | grep -qi nvidia; then' >>${tmp_script_file}
	echo '  echo "NVIDIA GPU detected, installing CUDA toolkit..."' >>${tmp_script_file}
	echo '  sudo apt install -y build-essential linux-headers-$(uname -r) wget ca-certificates' >>${tmp_script_file}
	echo '  ubuntu_ver=$(. /etc/os-release && echo ${VERSION_ID//./})' >>${tmp_script_file}
	echo '  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_ver}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb' >>${tmp_script_file}
	echo '  sudo dpkg -i /tmp/cuda-keyring.deb' >>${tmp_script_file}
	echo '  sudo apt update' >>${tmp_script_file}
	echo '  sudo apt install -y cuda-toolkit-12-6' >>${tmp_script_file}
	echo '  if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then' >>${tmp_script_file}
	echo '    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc' >>${tmp_script_file}
	echo '    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc' >>${tmp_script_file}
	echo '  fi' >>${tmp_script_file}
	echo 'else' >>${tmp_script_file}
	echo '  echo "No NVIDIA GPU detected, skipping CUDA install."' >>${tmp_script_file}
	echo 'fi' >>${tmp_script_file}

	# Send the script to all MACHINES via parallel SCP
	echo "Sending configuration script to ${MACHINES[*]}"
	for m in ${MACHINES[*]}; do
		scp ${tmp_script_file} ${USER}@${m}.${DOMAIN}:${config_command} &
	done
	wait
	rm ${tmp_script_file}

	# Use screen to run the script in parallel
	tmp_screen="$(mktemp)" || exit 1
	make_screen $tmp_screen
	for i in $(seq 0 ${last_valid_index}); do
		echo "screen -t node${i} ssh ${USER}@${MACHINES[$i]}.${DOMAIN} bash ${config_command}" >>${tmp_screen}
	done
	screen -c ${tmp_screen}
	rm ${tmp_screen}

	# Check the status of the IBV library on the target MACHINES
	check_ibv
	send_libs
}

function run_mu {
	# check if file exists
	EXE_NAME=$(basename "$1")
	if [[ ! -f "build/$1" ]]; then
		echo "Executable not found: $1"
		exit 1
	fi

	for m in ${MACHINES[*]}; do
		scp "build/$1" "${USER}@${m}.${DOMAIN}:${EXE_NAME}" &
	done

	wait
	rm -rf logs
	mkdir logs
	# Set up a screen script for running the program on all MACHINES
	tmp_screen="$(mktemp)" || exit 1
	make_screen "$tmp_screen"

	IDS=$(seq 1 $((${#MACHINES[@]})) | paste -sd, -)
	STARTING_PORT="6379"
	DORY_REGISTRY_IP="10.10.1.1:9999"
	NUM_MACHINES=${#MACHINES[@]}
	for i in "${!MACHINES[@]}"; do
		host="${MACHINES[$i]}"
		ENV_ARGS="EXPER_PORT=${STARTING_PORT} SID=$((i + 1)) IDS=${IDS} DORY_REGISTRY_IP=${DORY_REGISTRY_IP} LD_LIBRARY_PATH=~/"
		CMD="${ENV_ARGS} ./${EXE_NAME} --hostname ${host} --node-id ${i} --output-file mu_stats_${NUM_MACHINES}.csv ${ARGS}"
		echo "$CMD"
		cat >>"$tmp_screen" <<EOF
screen -t node${i} ssh ${USER}@${host}.${DOMAIN} ${CMD} ${EXTRA_ARGS}
logfile logs/log_${i}.txt
log on
EOF
	done

	screen -c "$tmp_screen"
	rm "$tmp_screen"
}

function run_mu_debug {
	# check if file exists
	EXE_NAME=$(basename "$1")
	if [[ ! -f "build/$1" ]]; then
		echo "Executable not found: $1"
		exit 1
	fi
	for m in ${MACHINES[*]}; do
		scp "build/$1" "${USER}@${m}.${DOMAIN}:${EXE_NAME}" &
	done
	wait
	rm -rf logs
	mkdir logs
	# Set up a screen script for running the program on all MACHINES
	tmp_screen="$(mktemp)" || exit 1
	make_screen "$tmp_screen"

	IDS=$(seq 0 $((${#MACHINES[@]} - 1)) | paste -sd, -)
	STARTING_PORT="6379"
	DORY_REGISTRY_IP="10.10.1.1:9999"
	GDB_ARGS=""
	for i in "${!MACHINES[@]}"; do
		if [[ $i -eq 0 ]]; then
			GDB_ARGS="gdb -ex \"catch throw\" -ex \"r\" --args"
		else
			GDB_ARGS=""
		fi
		host="${MACHINES[$i]}"
		ENV_ARGS="EXPER_PORT=${STARTING_PORT} SID=${i} IDS=${IDS} DORY_REGISTRY_IP=${DORY_REGISTRY_IP} LD_LIBRARY_PATH=~/"
		CMD="${ENV_ARGS} ${GDB_ARGS} ./${EXE_NAME} --hostname ${host} --node-id ${i} ${ARGS}"
		echo "$CMD"
		cat >>"$tmp_screen" <<EOF
screen -t node${i} ssh ${USER}@${host}.${DOMAIN} ${CMD} ${EXTRA_ARGS}; bash
logfile gdb-logs/gdb_${i}.log
log on
EOF
	done

	screen -c "$tmp_screen"
	rm "$tmp_screen"
}

function send_libs() {
	for m in ${MACHINES[*]}; do
		scp "lib/libcrashconsensus.so" "${USER}@${m}.${DOMAIN}:~/" &
		scp "lib/memcached" "${USER}@${m}.${DOMAIN}:~/" &
		scp "lib/libevent-2.1.so.6" "${USER}@${m}.${DOMAIN}:~/" &
		scp "lib/libcudart.so.12" "${USER}@${m}.${DOMAIN}:~/" &
	done
	wait
}

function reset-memcached() {
# Reset the memcached server
	ssh ${USER}@${MACHINES[0]}.${DOMAIN} "sudo pkill memcached"
	sleep 1
	# Launch the memcached server
	MEMCACHED_ARGS="-vv -p 9999"
	ssh ${USER}@${MACHINES[0]}.${DOMAIN} "nohup env LD_LIBRARY_PATH=/users/${USER} ./memcached ${MEMCACHED_ARGS} > memcached.log 2>&1 &"
}

function do_all {
	for i in "${!MACHINES[@]}"; do
		ssh ${USER}@${MACHINES[$i]}.${DOMAIN} "$1" &
	done
	wait
}

function reset-all() {
	last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes
	for i in $(seq 0 ${last_valid_index}); do
		ssh ${USER}@${MACHINES[$i]}.${DOMAIN} "sudo killall -9 -u $USER" &
	done
	wait
	echo "Nodes have been reset."
}

cmd="$1"
count=$#
EXE_PATH="src/smr"
cd $(git rev-parse --show-toplevel)
source config/cloudlab.conf

REMOTES=""
for machine in "${MACHINES[@]}"; do
	if [[ -z "$REMOTES" ]]; then
		REMOTES="$machine"
	else
		REMOTES="$REMOTES,$machine"
	fi
done
ARGS="--remotes ${REMOTES}"

if [[ "$cmd" == "build" && "$count" -eq 1 ]]; then
	cd ~/mu
	sudo docker run --privileged --rm -v $(pwd):/mu --name mu -it mu:latest
	bash transfer.sh
	cd ~/cas-paxos
elif [[ "$cmd" == "install-deps" && "$count" -eq 1 ]]; then
	install_deps
elif [[ "$cmd" == "run" && "$count" -eq 1 ]]; then
	run_mu "$EXE_PATH"
elif [[ "$cmd" == "reset" && "$count" -eq 1 ]]; then
	reset-all
elif [[ "$cmd" == "run-debug" && "$count" -eq 1 ]]; then
	run_mu_debug "$EXE_PATH"
elif [[ "$cmd" == "send-libs" && "$count" -eq 1 ]]; then
	send_libs
elif [[ "$cmd" == "reset-memcached" && "$count" -eq 1 ]]; then
	reset-memcached
elif [[ "$cmd" == "experiment" && "$count" -eq 2 ]]; then
	# expecting $2 to be 1 2 3 or 4
	# check to make sure $2 is valid by check > 1 and < 5
	if [[ "$2" -gt 0 && "$2" -lt 5 ]]; then
		# switch on $2
		case "$2" in
			1)
				source experiments/num_nodes.sh
				;;
			2)
				source experiments/matrix_size.sh
				;;
			3)
				source experiments/num_state_mat.sh
				;;
			4)
				source experiments/buf_sz.sh
				;;
		esac
	else
		echo "Usage: $0 experiment [1|2|3|4]"
		echo "1: Vary number of nodes"
		echo "2: Vary matrix size"
		echo "3: Vary number of state machines"
		echo "4: Vary buffer size"
		exit 1
	fi

else
	echo "Usage: $0 [build|run|reset|run-debug|send-libs|reset-memcached|experiment [1|2|3|4] ]"
	exit 1
fi
