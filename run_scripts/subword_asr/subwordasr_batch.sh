#!/bin/bash
#SBATCH --job-name=c_sasr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d
#SBATCH --array=2-5

LR="1.0e-${SLURM_ARRAY_TASK_ID}"
# LR="1.0e-${1}"
# EXP_MODEL_NAME="KW_bsz256_WS_integratedM_flickr_c0.5p0.5"
# EXP_MODEL_NAME="KW_bsz256_WS_multiAtt_c_flickr"
EXP_MODEL_NAME="KW_bsz256_WS_p1_flickr"

EXP_NAME="lr_${LR}"

CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_multiAtt_c_flickr/epoch=58-step=6902-val_recall_mean_1=7.7700.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch=248-step=29132-val_recall_mean_1=11.1800.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_c_transEnc_head1/epoch=332-step=38960-val_recall_mean_1=9.5300.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_cm_large/epoch=121-step=14273-val_recall_mean_1=12.5800.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_c_transEnc_head8/epoch=364-step=42704-val_recall_mean_1=10.9200.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_COCO_L14_HubertLarge_integrate_p0.5cm0.5/epoch=24-step=57824-val_recall_mean_1=5.8910.ckpt"

lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl

pwd

echo learning_rate=${LR}

# srun python testtest.py ${SLURM_ARRAY_TASK_ID}

srun python3 run_downstream.py -n $EXP_NAME \
    --expdir "result/downstream/subword_asr300/$EXP_MODEL_NAME/$EXP_NAME" \
    -m train \
    -u cascadedCLIP_local \
    -d ctc \
    -k $CKPT_PATH \
    -c downstream/ctc/subword_librispeech.yaml \
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

