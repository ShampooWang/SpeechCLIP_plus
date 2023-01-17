#!/bin/bash
#SBATCH --job-name=p4TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --account="MST111038"
#SBATCH --partition=gp4d
#SBATCH --output=slurms_log/train_Flickr_kwCLIPBase_p4TEST_slurm_%A


lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/atosystem/audio-visual-ssl

pwd


EXP_ROOT="exp_test/new_flickr/4layer/train_Flickr_kwCLIPBase_p4"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "/work/vjsalt22/atosystem/audio-visual-ssl/exp/4layer/train_Flickr_kwCLIPBase_p4/epoch=27-step=3275-val_recall_mean_10=77.4600.ckpt" \
    --gpus 8 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT
