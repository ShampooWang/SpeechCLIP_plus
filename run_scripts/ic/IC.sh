#!/bin/bash

LR="1.0e-$1"
# EXP_MODEL_NAME="train_Flickr_kwCLIP_multGPU_cm_large"

EXP_NAME="Flickr_integrated_normal2"
# CKPT_PATH="/work/twsezjg982/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"
# CKPT_PATH="/work/twsezjg982/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch=248-step=29132-val_recall_mean_1=11.1800.ckpt"

# CKPT_PATH="/work/b07901033/exp_LargeFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=187-step=21995-val_recall_mean_10=62.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_multiAtt_c_flickr/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_p1_flickr/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt"

#CKPT_PATH="/work/${CURRENT_USERNAME}/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_multiAtt_c_flickr/epoch=58-step=6902-val_recall_mean_1=7.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_p1_L14_HuLarge_s3prlNormed_learnableTemp_wavNorm/epoch_14-step_33224-val_recall_mean_10_84.0128.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_p_large/epoch=56-step=6668-val_recall_mean_10=89.0000.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_p_large/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_c2/epoch=162-step=19070-val_recall_mean_10=49.0600.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_p2/epoch=40-step=4796-val_recall_mean_10=78.7100.ckpt"
CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/integrated_normal/epoch=41-step=39353-val_recall_mean_10=50.5200.ckpt"

lscpu | egrep 'CPU\(s\)'
cd $s3prl
pwd

echo learning_rate=${LR}

# srun python testtest.py ${SLURM_ARRAY_TASK_ID}

python3 run_downstream.py -n $EXP_NAME \
    --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/IC/${EXP_NAME}/${LR}" \
    -m train -u cascadedCLIP_local \
    -d fluent_commands \
    -k $CKPT_PATH \
    -o "config.optimizer.lr=$LR"

# srun python3 run_downstream.py -n $EXP_NAME \
#     --expdir "result/downstream/IC/$EXP_NAME" \
#     -m train -u cascadedCLIP_local \
#     -d fluent_commands \
#     -k $CKPT_PATH \
#     -o "config.optimizer.lr=$LR"

# srun python3 run_downstream.py -n hubertBase_lr1e-4  -m train -u hubert_base  -d fluent_commands

# srun python3 run_downstream.py \z
#     -m train \
#     -e /work/twsezjg982/atosystem/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/cascadedCLIP_local_kw_eachbn_scale_1.0_cosineVq_heads_1_keyword_8_lr1e-5/states-200000.ckpt \
#     -o "config.runner.total_steps=300000"


# python3 run_downstream.py -n cascadedCLIP_local_fp16_1e-4_learnable_tmp_1.5 -m train -u cascadedCLIP_local  -d speech_translation -k /work/twsezjg982/atosytem/audio-visual-ssl/exp/kw_simple_fp16_lr1e-4/epoch=60-step=19093-val_recall_mean_1=27.8200.ckpt

