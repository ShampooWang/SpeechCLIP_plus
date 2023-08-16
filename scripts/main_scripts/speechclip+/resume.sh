#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md1/user_jeffwang/audio-visual-ssl

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --test \
    --resume "/mnt/md1/user_jeffwang/Checkpoints/speechclip+/hybrid/flickr/flickr_h/epoch=347-step=40716-val_recall_mean_10=40.1100.ckpt" \
    --save_path "/mnt/md1/user_jeffwang/Checkpoints/speechclip+/hybrid/flickr/flickr_h/test_p"
