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
cd ${s3prl}

# srun python testtest.py ${SLURM_ARRAY_TASK_ID}
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs/subword_asr1k/1.0e-4/states-130000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs/subword_asr300/1.0e-4/states-140000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs-plus/subword_asr1k/1.0e-4/states-129500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs-plus/subword_asr300/1.0e-4/states-131000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/hubert/lr_1.0e-4/states-66000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/hubert/lr_1.0e-4/states-66500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs/subword_asr300/1.0e-4/states-153500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/flickr_p_small_MHA/lr_1.0e-4/states-10500.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/flickr_p_small_MHA/lr_1.0e-4/dev-best.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/flickr_p_small_MHA/lr_1.0e-4/states-11000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/asr/parallelMHA/1.0e-4/states-38000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/PR/flickr_p_small_MHA/1.0e-4/states-37900.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/SF/flickr_p_large_MHA/1.0e-4/states-15000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/KWCOCO_KSplit_bsz256_1e4_integratedReal_c1.0_p1.0_L14_HuLarge_1024_1024_768_s3prlNormed/lr_1.0e-4/states-73000.ckpt"
# ckpt="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/flickr_p_small_MHA/lr_1.0e-4/states-20000.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/SF/Flickr_c+_small(cif)/1.0e-4/states-29500.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/asr/Flickr_c+_small(cif)/1.0e-4/states-1500.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_c+_small(cif)/1.0e-4/states-18400.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/SF/Flickr_c+_base_last_addcif/1.0e-4/states-3500.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/KS/Flickr_c+_small_dr20/1.0e-4/states-31000.ckpt"
# ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/subword_asr/Flickr_c+_small_dr20/1.0e-4/states-23500.ckpt"
ckpt="/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_h+_small_50k_all/1.0e-4/states-4400.ckpt"
python3 run_downstream.py -m train -e $ckpt

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

