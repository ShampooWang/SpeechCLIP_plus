#!/bin/bash
#SBATCH --job-name=c4TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --account="MST111038"
#SBATCH --partition=gp4d
#SBATCH --output=slurms_log/train_Flickr_kwCLIPBase_c4TEST_slurm_%A


lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/atosystem/audio-visual-ssl

pwd


EXP_ROOT="exp_test/new_flickr/4layer/train_Flickr_kwCLIPBase_c4"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_c4/epoch=42-step=5030-val_recall_mean_10=39.9800.ckpt" \
    --gpus 2 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT
