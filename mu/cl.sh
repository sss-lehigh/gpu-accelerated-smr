#!/bin/bash

# cl.sh
#
# A script for uploading and running Remus applications on CloudLab

set -e # Halt the script on any error

# Print usage information
usage() {
    cat <<EOF

Usage: cl.sh <command> [args]

Commands:
  install-deps
      Install dependencies onto the CloudLab machines
      (no additional args).
  build-run <mode>
      Build via build-cl.sh in the given <mode> and then run on CloudLab.
      <mode> must be one of:
        debug
        release
  run
      Run the previously built executable on CloudLab
      (no build step, no additional args).
  connect
      SSH/connect to the CloudLab machines for interactive debugging
      (no additional args).
  reset
      Kill all remote processes on the CloudLab machines
      (no additional args).

NOTE: Make sure your *.conf files are up to date before you run any command.

EOF
}

# SSH into MACHINES once, to fix known_hosts
function cl_first_connect {
    echo "Performing one-time connection to CloudLab MACHINES, to get known_hosts right"
    for machine in ${MACHINES[@]}; do
        ssh $USER@$machine.$DOMAIN echo "Connected"
    done
}

# Append the default configuration of a screenrc to the given file
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
        ssh $USER@$machine.$DOMAIN ibv_devinfo
    done
}

#  Configure the set of CloudLab MACHINES
function cl_install_deps() {
    config_command=prepare_to_run.sh          # The script to put on remote nodes
    last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes
    # First-time SSH
    cl_first_connect

    tmp_script_file="$(mktemp)" || exit 1
    echo 'hostname' >${tmp_script_file}
    echo 'sudo apt-get update && sudo apt-get -y install libibverbs-dev libmemcached-dev python3 python3-pip cmake ninja-build clang lld clang-format vim tmux git memcached libevent-dev libhugetlbfs-dev libgtest-dev libnuma-dev numactl libgflags-dev' >${tmp_script_file}

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
        cat >>"$tmp_screen" <<EOF
screen -t node${i} ssh ${USER}@${MACHINES[$i]}.${DOMAIN} bash ${config_command}
logfile logs/log_${i}.txt
log on
EOF
    done
    screen -c ${tmp_screen}
    rm ${tmp_screen}
}

# SEND and RUN a binary on the CloudLab MACHINES
function cl_run() {
    for m in ${MACHINES[*]}; do
        scp "build/${EXE_NAME}" "${USER}@${m}.${DOMAIN}:${EXE_NAME}" &
    done
    wait
    rm -rf logs
    mkdir logs

    echo "Args: ${ARGS}"
    # Set up a screen script for running the program on all MACHINES
    tmp_screen="$(mktemp)" || exit 1
    make_screen "$tmp_screen"

    for i in "${!MACHINES[@]}"; do
        host="${MACHINES[$i]}"
        cat >>"$tmp_screen" <<EOF
screen -t node${i} ssh ${USER}@${host}.${DOMAIN} ./${EXE_NAME} --node-id ${i} ${ARGS}; bash
logfile logs/log_${i}.txt
log on
EOF
    done

    screen -c "$tmp_screen"
    rm "$tmp_screen"
}

# Connect to CloudLab nodes (e.g., for debugging)
function cl_connect() {
    last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes

    # Set up a screen script for connecting
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    for i in $(seq 0 ${last_valid_index}); do
        cat >>"$tmp_screen" <<EOF
screen -t node${i} ssh ${USER}@${MACHINES[$i]}.${DOMAIN}
logfile logs/log_${i}.txt
log on
EOF
    done
    screen -c $tmp_screen
    rm $tmp_screen
}

function reset() {
    last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes

    for i in $(seq 0 ${last_valid_index}); do
        ssh ${USER}@${MACHINES[$i]}.${DOMAIN} "pkill ${EXE_NAME}; rm -rf ~/*"
    done
    echo "Nodes have been reset."
}

function retrieve_results {
    echo "Retrieving results..."
    last_valid_index=$((${#MACHINES[@]} - 1)) # The 0-indexed number of nodes
    mkdir -p results
    for i in $(seq 0 ${last_valid_index}); do
        host="${MACHINES[$i]}"
        scp "${USER}@${host}.${DOMAIN}:~/results.csv" "results/metrics_${host}.csv"
    done
}

# Get the important stuff out of the command-line args
cmd=$1   # The requested command
count=$# # The number of command-line args
# Navigate the the project root directory
cd $(git rev-parse --show-toplevel)
# Load the config right away
source cl.config

if [[ "$cmd" == "install-deps" && "$count" -eq 1 ]]; then
    cl_install_deps
elif [[ "$cmd" == "build-run" && "$count" -eq 2 ]]; then
    if [[ "$2" != "debug" && "$2" != "release" ]]; then
        usage
        exit 1
    fi
    source tools/build-cl.sh "$2"
    cl_run
    retrieve_results
elif [[ "$cmd" == "run" && "$count" -eq 1 ]]; then
    cl_run
    retrieve_results
elif [[ "$cmd" == "connect" && "$count" -eq 1 ]]; then
    cl_connect
elif [[ "$cmd" == "reset" && "$count" -eq 1 ]]; then
    reset
else
    usage
fi
