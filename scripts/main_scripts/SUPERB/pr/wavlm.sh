#!/bin/bash
#SBATCH --job-name=w2vuPool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111038"
#SBATCH --partition=gp4d
#SBATCH --out=logs/superb/train/PR_wavlm_base.log

lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl
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
ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_c2_small/epoch=99-step=46799-val_recall_mean_10=42.2300.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/Flickr_p2_small/epoch=20-step=9827-val_recall_mean_10=77.7200.ckpt"
LR="1.0e-${1}"
echo ${LR}

exp_name="wavlm_base"

model_class=wavlm_base # wavlm_base_plus wavlm_large
# python3 run_downstream.py -n ExpName -m train -u fbank -d ctc -c downstream/ctc/snips.yaml

python3 run_downstream.py -n PR_$exp_name \
        --expdir "/work/vjsalt22/SUPERB/wavlm/PR/$exp_name/$LR" \
        -m train \
        -u $model_class \
        -d ctc \
        -c downstream/ctc/libriphone.yaml \
        -o "\
        config.optimizer.lr=$LR,,
        config.downstream_expert.corpus.path='/work/vjsalt22/dataset/LibriSpeech100' \
        "
