#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

# EXP_ROOT="/mnt/md0/user_jeff/audio-visual-ssl/exp/2layer/integrated_normal"
EXP_ROOT="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/integrated_normal"

# CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/KWClip/train_flickr_kwCLIP_integrated.yaml"
CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/KWClip/wavlm/Flickr_h2_small.yaml"


mkdir $EXP_ROOT

# srun python3 run_task.py \
#     "TrainKWClip_GeneralTransformer" \
#     --config "config/KWClip/train_flickr_kwCLIP.yaml" \
#     --device "cuda:0" \
#     --gpus 1 \
#     --njobs 8 \
#     --seed 7122 \
#     --train \
#     --save_path $EXP_ROOT

# srun python3 run_task.py \
#     "TrainKWClip_GeneralTransformer" \
#     --config "config/KWClip/train_flickr_kwCLIP_multGPU.yaml" \
#     --gpus 8 \
#     --njobs 8 \
#     --seed 7122 \
#     --train \
#     --save_path $EXP_ROOT

python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG_FILE \
    --gpus 3 \
    --njobs 3 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
