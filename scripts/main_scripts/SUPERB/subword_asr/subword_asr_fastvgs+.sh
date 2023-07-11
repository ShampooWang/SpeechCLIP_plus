#!/bin/bash
#SBATCH --job-name=cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d

lscpu | egrep 'CPU\(s\)'

cd ${s3prl}

LR="1.0e-${1}"
echo ${LR}

exp_name="fast-vgs-plus"
# exp_name="hubert"
# exp_name="hubert_large_ll60k"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small_test_150k_1e-4_pad/epoch=314-step=147419-val_recall_mean_10=30.4300.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_h+_small/epoch=48-step=22931-val_recall_mean_10=25.8800.ckpt"
# ckpt_path="${h50k}"
# exp_name="Flickr_h+_small_50k_all"

# model_class="cascadedCLIP_local"
model_class="fast_vgs"

# python3 run_downstream.py -n SUBWORD_ASR_$exp_name \
#         --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/subword_asr/$exp_name/lr_$LR" \
#         -m train \
#         -u $model_class \
#         -d ctc \
#         -c downstream/ctc/subword_librispeech.yaml \
#         -k $ckpt_path \
#         -o "config.optimizer.lr=$LR"

# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=185-step=21761-val_recall_mean_10=63.4800.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_small_MHA/epoch=122-step=14390-val_recall_mean_10=79.0200.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_large_MHA/epoch=41-step=4913-val_recall_mean_10=88.7800.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_integrated_large/epoch_118-step_13922-val_recall_mean_10_41.1000.ckpt"
# python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_p1_flickr/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small/epoch=263-step=61775-val_kw_hit_rate=60.1851.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/upstream/fast-vgs/weights/fast-vgs-plus-coco"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_multiAtt_c_flickr/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_p2_small/epoch=7-step=3743-val_recall_mean_10=77.9900.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl/upstream/fastvgs/weights/fast-vgs-plus-coco"
ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl/upstream/fastvgs/weights/fast-vgs-coco"


python3 run_downstream.py -n SUBWORD_ASR_$exp_name \
        --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/subword_asr/$exp_name/$LR" \
        -m train \
        -u $model_class \
        -k $ckpt_path \
        -d ctc \
        -c downstream/ctc/subword_librispeech.yaml \
        -o "config.optimizer.lr=$LR"


