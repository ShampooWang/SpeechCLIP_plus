#!/bin/bash

lscpu | egrep 'CPU\(s\)'

cd ${s3prl}

# exp_name="1e-4_kw_lr_1e-3_heads_1_keyword_8"

model_class="cascadedCLIP_local"
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/asr/train_Flickr_kwCLIP_multGPU_integrated_large/1.0e-4/dev-clean-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/asr/parallelMHA/1.0e-4/dev-clean-best.ckpt"
# ckpt_path="/mnt/md0/user_jeff/SUPERB/wavlm/asr/Flickr_c+_small(cif)/1.0e-4/dev-clean-best.ckpt"

python3 run_downstream.py -m evaluate -t "test-clean" -e $ckpt_path