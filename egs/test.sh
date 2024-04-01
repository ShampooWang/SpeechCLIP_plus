#!/bin/bash
lscpu | egrep 'CPU\(s\)'
cd /mnt/md1/user_jeffwang/SpeechCLIP-plus
EXP_ROOT="/mnt/md1/user_jeffwang/SpeechCLIP-plus/test"
DATASET_ROOT="/mnt/md1/user_jeffwang/Dataset/flickr"


python run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "/mnt/md1/user_jeffwang/SpeechCLIP-plus/icassp_sasb_ckpts/SpeechCLIP+/base/flickr/cascaded+/model.ckpt" \
    --dataset_root ${DATASET_ROOT} \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --test \
    --save_path ${EXP_ROOT}
    

