#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

EXP_ROOT="/mnt/md0/user_jeff/Checkpoints/speechclip+/parallel/flickr/flickr_p"
CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/speechclip+/parallel/flickr/flickr_p2.yaml"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --config $CFG_FILE \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT