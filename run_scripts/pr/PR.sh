#!/bin/bash
#SBATCH --job-name=PRFc2l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111038"
#SBATCH --partition=gp4d
#SBATCH --out=logs/superb/train/PR_Flickr_c2_large.log

lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl/tmp_tools/s3prl/s3prl
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/exp/KW_bsz256_WS_c1Mp1_flickr_c0.5p0.5/epoch=241-step=28313-val_recall_mean_1=9.5100.ckpt"
#ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/LargeOnFlickr/train_Flickr_kwCLIP_multGPU_cm_large/epoch=185-step=21761-val_recall_mean_10=63.4800.ckpt"

# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_small_MHA/epoch=122-step=14390-val_recall_mean_10=79.0200.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/parallelMHA/flickr_p_large_MHA/epoch=41-step=4913-val_recall_mean_10=88.7800.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_c2/epoch=162-step=19070-val_recall_mean_10=49.0600.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/2layer/train_Flickr_kwCLIPBase_p2/epoch=40-step=4796-val_recall_mean_10=78.7100.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_c4/epoch=42-step=5030-val_recall_mean_10=39.9800.ckpt"
# ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_p4/epoch=27-step=3275-val_recall_mean_10=77.4600.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/2layer/train_COCO_kwCLIPBase_p2/epoch=13-step=31009-val_recall_mean_10=78.0333.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/2layer/train_COCO_kwCLIPBase_c2/epoch=11-step=26579-val_recall_mean_10=43.6506.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/4layer/train_COCO_kwCLIPBase_p4/epoch=10-step=24364-val_recall_mean_10=78.5531.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/4layer/train_COCO_kwCLIPBase_c4/epoch=10-step=25992-val_recall_mean_10=45.5640.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=32-step=15443-val_recall_mean_10=40.6600.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=99-step=46799-val_recall_mean_10=42.2300.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_p2_small/epoch=20-step=9827-val_recall_mean_10=77.7200.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_c2_small/epoch=22-step=49999-val_recall_mean_10=34.0082.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_small/epoch=22-step=49999-val_recall_mean_10=71.7749.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/full_subword_vocab/COCO_c2_small/epoch=16-step=39320-val_recall_mean_10=27.9004.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_c2_large/epoch=7-step=18503-val_recall_mean_10=24.7825.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_large/epoch=15-step=37007-val_recall_mean_10=75.7473.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_p2_large/epoch=18-step=2222-val_recall_mean_10=86.7500.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_small(ks_split)/epoch=20-step=46514-val_recall_mean_10=71.9549.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_large/epoch=109-step=12869-val_recall_mean_10=54.3800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c2_large/epoch=109-step=12869-val_recall_mean_10=54.3800.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/2layer/integrated/epoch=53-step=49999-val_recall_mean_10=46.0200.ckpt"
# ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/integrated_normal/epoch=41-step=39353-val_recall_mean_10=50.5200.ckpt"
ckpt_path="/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/Flickr_c+_small(cif)/epoch=18-step=17802-val_kw_hit_rate=20.7648.ckpt"

LR="1.0e-${1}"
echo ${LR}

exp_name="Flickr_c+_small(cif)"

model_class="cascadedCLIP_local"


# python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml
python3 run_downstream.py -n PR_$exp_name \
        --expdir "${work}/SUPERB/wavlm/PR/$exp_name/$LR" \
        -m train \
        -u $model_class \
        -d ctc \
        -k $ckpt_path \
        -c downstream/ctc/libriphone.yaml \
        -o "\
        config.optimizer.lr=$LR,,
        config.downstream_expert.corpus.path='/mnt/md0/dataset/LibriSpeech100' \
        "
