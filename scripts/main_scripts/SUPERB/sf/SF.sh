#!/bin/bash


lscpu | egrep 'CPU\(s\)'
# cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl
cd ${s3prl}
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_c1Mp1_flickr_c0.5p0.5/epoch=241-step=28313-val_recall_mean_1=9.5100.ckpt"
#ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=185-step=21761-val_recall_mean_10=63.4800.ckpt"

# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_small_MHA/epoch=122-step=14390-val_recall_mean_10=79.0200.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/integrated_normal/epoch=41-step=39353-val_recall_mean_10=50.5200.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_large_MHA/epoch=41-step=4913-val_recall_mean_10=88.7800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=18-step=17802-val_kw_hit_rate=20.7648.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_h+_small/epoch=12-step=6083-val_kw_hit_rate=19.1170.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_h2_base/epoch=107-step=25271-val_recall_mean_10=48.3800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small/epoch=263-step=61775-val_kw_hit_rate=60.1851.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_h2_base_reg_sa/epoch=19-step=37499-val_recall_mean_10=45.8200.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_base_reg/epoch=71-step=134999-val_recall_mean_10=30.3800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_p2_small/epoch=7-step=3743-val_recall_mean_10=77.9900.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_base_last/epoch=248-step=116531-val_recall_mean_10=31.4100.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small_test_150k_1e-4_pad/epoch=314-step=147419-val_recall_mean_10=30.4300.ckpt"
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_h+_small/epoch=48-step=22931-val_recall_mean_10=25.8800.ckpt"
ckpt_path="${h50k}"

LR="1.0e-${1}"
echo ${LR}

exp_name="COCO_SpeechCLIP_p2_all"

model_class="cascadedCLIP_local"

# python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml
python3 run_downstream.py -n SF_$exp_name \
        --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/SF/$exp_name/$LR" \
        -m train \
        -u $model_class \
        -d ctc \
        -c downstream/ctc/snips.yaml \
        -k "${cp}" \
        -o "config.optimizer.lr=$LR"
