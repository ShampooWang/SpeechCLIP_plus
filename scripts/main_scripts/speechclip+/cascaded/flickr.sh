#!/bin/bash
#SBATCH --job-name=flickr_kwc_large_fairseq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/SpeechCLIP_plus/logs/speechclip+/cascaded/flickr_kwc_large_fairseq.log

lscpu | egrep 'CPU\(s\)'
cd /work/jgtf0322/SpeechCLIP_plus

EXP_ROOT="/work/jgtf0322/SpeechCLIP_plus/checkpoints/speechclip+/cascaded/flickr/flickr_kwc_large_fairseq"
CFG_FILE="/work/jgtf0322/SpeechCLIP_plus/config/speechclip+/cascaded/flickr/flickr_c2.yaml"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --config $CFG_FILE \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT