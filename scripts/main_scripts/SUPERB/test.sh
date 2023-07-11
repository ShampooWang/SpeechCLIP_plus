#!/bin/bash
lscpu | egrep 'CPU\(s\)'

cd /mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl

############## Gerneral ###############
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/SUPERB/asr/topline_wo_attn_lr2e-5_cos_wo_dcl/1.0e-4/dev-clean-best.ckpt"
# python3 run_downstream.py -m evaluate -e $ckpt_path
#######################################


############### ASR ###################
ckpt_path="/mnt/md0/user_jeff/Checkpoints/SUPERB/asr/topline_wo_attn_lr2e-5_cos_wo_dcl/1.0e-4/dev-clean-best.ckpt"
python3 run_downstream.py -m evaluate -t "test-clean" -e $ckpt_path
#######################################


############### ER ####################
# for test_fold in fold1 fold2 fold3 fold4 fold5;
# do
#     # The default config is "downstream/emotion/config.yaml"
#     # python3 run_downstream.py -n ExpName_$test_fold -m train -u fbank -d emotion -o "config.downstream_expert.datarc.test_fold='$test_fold'"
#     python3 run_downstream.py -m evaluate -e "/work/vjsalt22/result/downstream/ER/train_Flickr_p2_chimera_chimera_setup2_2_dis_t_cont_init_mel_tr3_${test_fold}/dev-best.ckpt"
# done
#######################################

############## Change Upstream and dataset path###############
# ckpt_path="/work/vjsalt22/result/downstream/KS/train_COCO_kwCLIPBase_c4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/Flickr_wavlmp2_small_nonormalize/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/Flickr_wavlmp2_small/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/wavlm_base/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/wavlm_base/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/COCO_wavlmc2_small/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/COCO_wavlmp2_small/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/SUPERB/wavlm/PR/COCO_wavlmc2_small_fullvocab/1.0e-4/dev-best.ckpt"
# ckpt_path="/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_c2_small(base+)/1.0e-4/dev-best.ckpt"
# ckpt_path="/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_p2_small(base+)/1.0e-4/dev-best.ckpt"
# ckpt_path="/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_integrated2/1.0e-4/dev-best.ckpt"
# python3 run_downstream.py -m evaluate -e $ckpt_path \
#             -o "\
#             config.downstream_expert.datarc.libri_root='/mnt/md0/dataset/LibriSpeech100',, \
#             config.downstream_expert.corpus.path='/mnt/md0/dataset/LibriSpeech100',, \
#             args.expdir='/mnt/md0/user_jeff/SUPERB/wavlm/PR/Flickr_p2_small(base+)/1.0e-4',, \
#             args.upstream_ckpt='/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_p2_small(base+)/epoch_44-step_5264-val_recall_mean_10_79.5500.ckpt'
#             "
#######################################

