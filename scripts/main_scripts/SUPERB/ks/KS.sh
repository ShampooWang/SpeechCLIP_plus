#!/bin/bash

lscpu | egrep 'CPU\(s\)'

cd /mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl

ckpt="/mnt/md0/user_jeff/Checkpoints/speechclip+/topline_wo_attn_lr2e-5_cos_wo_dcl/epoch=45-step=38318-val_recall_mean_10=38.6800.ckpt"

LR="1.0e-4"
EXP_NAME="topline_wo_attn_lr2e-5_cos_wo_dcl"
model_class="customized_upstream"

echo learning_rate=${LR}
echo $EXP_NAME

python3 run_downstream.py -n $EXP_NAME \
    --expdir "/mnt/md0/user_jeff/Checkpoints/SUPERB/ks/${EXP_NAME}/${LR}" \
    -m train -u customized_upstream \
    -d speech_commands \
    -k "${ckpt}" \
    -o "config.optimizer.lr=$LR" \