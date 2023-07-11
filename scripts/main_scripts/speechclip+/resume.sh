#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --train \
    --resume /mnt/md0/user_jeff/Checkpoints/speechclip+/cascaded/flickr/flickr_c+_w_cos_pen/last.ckpt
