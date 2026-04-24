EXE_PATH="experiments/caspaxos"

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

function reset_mu() {
	for m in ${MACHINES[*]}; do
		scp "lib/libcrashconsensus.so" "${USER}@${m}.${DOMAIN}:libcrashconsensus.so" &
	done
	
	MISSING_FILES=$(ssh ${USER}@${MACHINES[0]}.${DOMAIN} "test -f ~/memcached && test -f ~/libevent-2.1.so.6 && echo false || echo true")


	if [[ "$MISSING_FILES" == "true" ]]; then
		echo "Critical files do not exist on remote. Sending over now..."
		# Set up memcached on node0
		scp "lib/memcached" "${USER}@${MACHINES[0]}.${DOMAIN}:memcached"
		scp "lib/libevent-2.1.so.6" "${USER}@${MACHINES[0]}.${DOMAIN}:~/"
	fi

	# Reset the memcached server
	ssh ${USER}@${MACHINES[0]}.${DOMAIN} "sudo pkill memcached"
	sleep 1
	# Launch the memcached server
	MEMCACHED_ARGS="-vv -p 9999"
	ssh ${USER}@${MACHINES[0]}.${DOMAIN} "nohup env LD_LIBRARY_PATH=/users/${USER} ./memcached ${MEMCACHED_ARGS} > memcached.log 2>&1 &"
}

cmd="$1"
count=$#

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
	reset_mu
elif [[ "$cmd" == "run-debug" && "$count" -eq 1 ]]; then
	run_mu_debug "$EXE_PATH"
else
	echo "Usage: $0 [build|run|reset|run-debug]"
	exit 1
fi
