#!/bin/bash
#SBATCH --job-name=asr_c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d

lscpu | egrep 'CPU\(s\)'

cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl

# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"
#ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=185-step=21761-val_recall_mean_10=63.4800.ckpt"


# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_small_MHA/epoch=122-step=14390-val_recall_mean_10=79.0200.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_large_MHA/epoch=41-step=4913-val_recall_mean_10=88.7800.ckpt"
ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_p1_L14_HuLarge_s3prlNormed_learnableTemp_wavNorm/epoch_14-step_33224-val_recall_mean_10_84.0128.ckpt"
# LR="1.0e-${1}"
# echo ${LR}

exp_name="hubert_large"

model_class="cascadedCLIP_local"

python3 run_downstream.py -n ASR_$exp_name \
        --expdir "result/downstream/L2_norm/$exp_name/$LR" \
        -t 'test-clean' \
        -m get_features_norm \
        -u hubert_large_ll60k \
        -d asr \
        -c downstream/asr/config.yaml \
        -o "\
        config.downstream_expert.datarc.libri_root='/work/vjsalt22/dataset/LibriSpeech100' \
        "
