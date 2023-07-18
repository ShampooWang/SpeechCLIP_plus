#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --gpus 3 \
    --njobs 8 \
    --seed 7122 \
    --test \
    --resume "/mnt/md0/user_jeff/Checkpoints/speechclip+/cascaded/flickr/Flickr_SpeechCLIP_c2/epoch=80-step=37907-val_recall_mean_10=40.5200.ckpt"
