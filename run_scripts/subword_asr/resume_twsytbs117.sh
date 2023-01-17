#!/bin/bash
#SBATCH --job-name=SF_resume_pcoco
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d

# LR="1.0e-1"
# EXP_MODEL_NAME="KW_bsz256_WS_integratedM_flickr_c0.5p0.5"
# EXP_MODEL_NAME="KW_bsz256_WS_p1_flickr"

# EXP_NAME="KWCLIP_${EXP_MODEL_NAME}_lr_${LR}"

# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch=248-step=29132-val_recall_mean_1=11.1800.ckpt"
# CKPT_PATH="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_c_transEnc_head1/epoch=332-step=38960-val_recall_mean_1=9.5300.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"

lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl

# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/lr_1.0e-4/states-72500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/KWCOCO_KSplit_bsz256_1e4_integratedReal_c1.0_p1.0_L14_HuLarge_1024_1024_768_s3prlNormed/lr_1.0e-4/states-74500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/KWCOCO_KSplit_bsz256_1e4_integratedReal_c1.0_p1.0_L14_HuLarge_1024_1024_768_s3prlNormed/lr_1.0e-4/states-73000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/train_Flickr_kwCLIP_multGPU_cm_large/lr_1.0e-4/states-73000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/train_Flickr_kwCLIP_multGPU_cm_large/lr_1.0e-4/states-71500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/train_Flickr_kwCLIP_multGPU_mixed_large/lr_1.0e-4/states-67000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/train_Flickr_kwCLIP_multGPU_mixed_large/lr_1.0e-4/states-68500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/train_Flickr_kwCLIP_multGPU_integrated_large/lr_1.0e-4/states-72500.ckpt"
ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/train_Flickr_kwCLIP_multGPU_integrated_large/lr_1.0e-4/states-75000.ckpt"

# upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt"
# upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_integratedReal_c1.0_p1.0_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_21-step_48729-val_recall_mean_10_25.4413.ckpt"
# upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt"
# upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_p_large/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt"
upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_integrated_large/epoch_118-step_13922-val_recall_mean_10_41.1000.ckpt"

python3 run_downstream.py -m train -e $ckpt \
        -o "\
        config.downstream_expert.corpus.path='/work/vjsalt22/dataset/LibriSpeech100',, \
        args.upstream_ckpt=${upstream_ckpt_path}
        "

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

