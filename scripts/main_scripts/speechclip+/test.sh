#!/bin/bash
#SBATCH --job-name=devTest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/SpeechCLIP_plus/logs/speechclip+/test/devTest.log


lscpu | egrep 'CPU\(s\)'
cd /work/jgtf0322/SpeechCLIP_plus


python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --test \
    --resume "/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/large/flickr/cascaded/epoch_187-step_21995-val_recall_mean_10_62.7700.ckpt" \
    --save_path "/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/large/flickr/cascaded/test"