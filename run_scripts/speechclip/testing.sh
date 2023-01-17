#!/bin/bash
#SBATCH --job-name=FkrSmallC4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --account="MST111038"
#SBATCH --partition=gp2d
#SBATCH --output=logs/speechclip/test.log

lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/hsuanfu/audio-visual-ssl

pwd

EXP_ROOT="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/test"

CFG_FILE="config/KWClip/wavlm/test.yaml"


mkdir $EXP_ROOT

# srun python3 run_task.py \
#     "TrainKWClip_GeneralTransformer" \
#     --config "config/KWClip/train_flickr_kwCLIP.yaml" \
#     --device "cuda:0" \
#     --gpus 1 \
#     --njobs 8 \
#     --seed 7122 \
#     --train \
#     --save_path $EXP_ROOT

# srun python3 run_task.py \
#     "TrainKWClip_GeneralTransformer" \
#     --config "config/KWClip/train_flickr_kwCLIP_multGPU.yaml" \
#     --gpus 8 \
#     --njobs 8 \
#     --seed 7122 \
#     --train \
#     --save_path $EXP_ROOT

python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG_FILE \
    --gpus 2 \
    --njobs 2 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
