#!/bin/bash

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

EXP_ROOT="/mnt/md0/user_jeff/Checkpoints/COCO_SpeechCLIP_c2_freeze"

# CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/KWClip/train_flickr_kwCLIP_integrated.yaml"
CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/KWClip/wavlm/COCO_c2_small.yaml"


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
    --gpus 2 \
    --njobs 2 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
