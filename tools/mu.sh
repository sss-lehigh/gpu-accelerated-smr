

function make_screen {
	echo 'startup_message off' >>$1
	echo 'defscrollback 10000' >>$1
	echo 'autodetach on' >>$1
	echo 'escape ^jj' >>$1
	echo 'defflow off' >>$1
	echo 'hardstatus alwayslastline "%w"' >>$1
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
screen -t node${i} ssh ${USER}@${host}.${DOMAIN} ${CMD}
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
screen -t node${i} ssh ${USER}@${host}.${DOMAIN} ${CMD}; bash
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
else
	echo "Usage: $0 [build|run|reset|run-debug|send-libs|reset-memcached]"
	exit 1
fi
