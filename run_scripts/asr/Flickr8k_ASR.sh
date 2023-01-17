#!/bin/bash
#SBATCH --job-name=asr_c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d

lscpu | egrep 'CPU\(s\)'

cd ${s3prl}

# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"
#ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=185-step=21761-val_recall_mean_10=63.4800.ckpt"


# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_small_MHA/epoch=122-step=14390-val_recall_mean_10=79.0200.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_large_MHA/epoch=41-step=4913-val_recall_mean_10=88.7800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=18-step=17802-val_kw_hit_rate=20.7648.ckpt"
ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=18-step=17802-val_kw_hit_rate=20.7648.ckpt"


LR="1.0e-${1}"
echo ${LR}

exp_name="wavlm_base_plus" # Flickr_c+_small(cif)

model_class="cascadedCLIP_local"

python3 run_downstream.py -n ASR_$exp_name \
        --expdir "/mnt/md0/user_jeff/SUPERB/Flickr8k/wavlm/normalized_text_asr/$exp_name/$LR" \
        -m train \
        -u wavlm_base_plus \
        -d ctc \
        -c downstream/ctc/flickr8k_char.yaml \
        -k $ckpt_path \
        -o "config.optimizer.lr=$LR"
