#!/bin/bash

lscpu | egrep 'CPU\(s\)'

cd /mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl

ckpt="/mnt/md0/user_jeff/Checkpoints/speechclip+/topline_wo_attn_lr2e-5_cos_wo_dcl/epoch=45-step=38318-val_recall_mean_10=38.6800.ckpt"

LR="1.0e-4"
echo ${LR}

exp_name="topline_wo_attn_lr2e-5_cos_wo_dcl"

model_class="customized_upstream"

python3 run_downstream.py -n ASR_$exp_name \
        --expdir "/mnt/md0/user_jeff/Checkpoints/SUPERB/asr/$exp_name/$LR" \
        -m train \
        -u $model_class \
        -d asr \
        -c downstream/asr/config.yaml \
        -k "${ckpt}" \
        -o "\
        config.optimizer.lr=$LR,,
        config.downstream_expert.datarc.libri_root='/mnt/md0/dataset/LibriSpeech100' \
        "
