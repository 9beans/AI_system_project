#!/bin/bash

CONFIG_FILE="./configs/TPUv4.cfg"
CONFIG_NAME="TPUv4"
PE_SIZE=16384
PE_THRESHOLD=32
MEM_SIZE="6144,6144,4096"
BANDWIDTH="48"
MODE="USER"

mkdir -p "./Logs"

LOG_FILE="./Logs/$CONFIG_NAME-$PE_SIZE-$PE_THRESHOLD-$MEM_SIZE-$MODE.log"

python -u auto_config_generator.py \
    -c $CONFIG_FILE \
    -s $PE_SIZE \
    -t $PE_THRESHOLD \
    -m $MEM_SIZE \
    -b $BANDWIDTH \
    -e $MODE | tee "$LOG_FILE"
