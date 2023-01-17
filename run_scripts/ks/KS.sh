#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gp4d
#SBATCH --out=logs/superb/train/test.log

LR="1.0e-$1"
EXP_NAME="Flickr_c+_small_dr20_catkw"

# CKPT_PATH="/work/twsezjg982/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_p1_flickr/epoch=131-step=15443-val_recall_mean_1=36.0100.ckpt"
# CKPT_PATH="/work/twsezjg982/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch=248-step=29132-val_recall_mean_1=11.1800.ckpt"

# CKPT_PATH="/work/b07901033/exp_LargeFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=187-step=21995-val_recall_mean_10=62.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_multiAtt_c_flickr/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_p1_flickr/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt"
#CKPT_PATH="/work/${CURRENT_USERNAME}/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_multiAtt_c_flickr/epoch=58-step=6902-val_recall_mean_1=7.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_cm_L14_HuLarge_1024_1024_768_s3prlNormed/epoch_12-step_28794-val_recall_mean_10_36.1455.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_p1_L14_HuLarge_s3prlNormed_learnableTemp_wavNorm/epoch=14-step=33224-val_recall_mean_10=84.0128.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/COCO/KWCOCO_KSplit_bsz256_1e4_p1_L14_HuLarge_s3prlNormed_learnableTemp_wavNorm/epoch_14-step_33224-val_recall_mean_10_84.0128.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_p_large/epoch=56-step=6668-val_recall_mean_10=89.0000.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_p_large/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_c2/epoch=162-step=19070-val_recall_mean_10=49.0600.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_p2/epoch=40-step=4796-val_recall_mean_10=78.7100.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_p4/epoch=27-step=3275-val_recall_mean_10=77.4600.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_c4/epoch=42-step=5030-val_recall_mean_10=39.9800.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/2layer/train_COCO_kwCLIPBase_c2/epoch=11-step=26579-val_recall_mean_10=43.6506.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/2layer/train_COCO_kwCLIPBase_p2/epoch=13-step=31009-val_recall_mean_10=78.0333.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/4layer/train_COCO_kwCLIPBase_p4/epoch=10-step=24364-val_recall_mean_10=78.5531.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/4layer/train_COCO_kwCLIPBase_c4/epoch=10-step=25992-val_recall_mean_10=45.5640.ckpt"
# CKPT_PATH="/work/vjsalt22/atosystem/audio-visual-ssl/exp/chimera/train_Flickr_p2_chimera_chimera_setup2_2_dis_t_cont_init_mel_tr3/epoch=16-step=1988-val_recall_mean_10=7.3100.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=32-step=15443-val_recall_mean_10=40.6600.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=99-step=46799-val_recall_mean_10=42.2300.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_p2_small/epoch=20-step=9827-val_recall_mean_10=77.7200.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_c2_small/epoch=22-step=49999-val_recall_mean_10=34.0082.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_small/epoch=22-step=49999-val_recall_mean_10=71.7749.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/full_subword_vocab/COCO_c2_small/epoch=16-step=39320-val_recall_mean_10=27.9004.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_c2_large/epoch=7-step=18503-val_recall_mean_10=24.7825.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_large/epoch=15-step=37007-val_recall_mean_10=75.7473.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_large/epoch=109-step=12869-val_recall_mean_10=54.3800.ckpt"
# CKPT_PATH="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_p2_large/epoch=18-step=2222-val_recall_mean_10=86.7500.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=99-step=46799-val_recall_mean_10=42.2300.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/2layer/integrated/epoch=53-step=49999-val_recall_mean_10=46.0200.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/integrated_normal/epoch=41-step=39353-val_recall_mean_10=50.5200.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small/epoch=7-step=7495-val_recall_mean_10=29.3100.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=18-step=17802-val_kw_hit_rate=20.7648.ckpt"
# CKPT_PATH="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=20-step=19676-val_recall_mean_10=27.1800.ckpt"
CKPT_PATH="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small_test_150k_1e-4_pad/epoch=314-step=147419-val_recall_mean_10=30.4300.ckpt"

lscpu | egrep 'CPU\(s\)'
# cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl
# cd /work/vjsalt22/atosystem/audio-visual-ssl/tmp_tools/s3prl/s3prl
cd $s3prl

echo learning_rate=${LR}
echo $EXP_NAME

python3 run_downstream.py -n $EXP_NAME \
    --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/KS/${EXP_NAME}/${LR}" \
    -m train -u cascadedCLIP_local \
    -d speech_commands \
    -k $CKPT_PATH \
    -o "config.optimizer.lr=$LR"