#!/bin/bash
#SBATCH --job-name=FkrSmallP4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --account="MST111038"
#SBATCH --partition=gp4d
#SBATCH --output=/work/vjsalt22/logs/speechclip/train/COCO_p2_large.log

lscpu | egrep 'CPU\(s\)'
cd /work/vjsalt22/hsuanfu/audio-visual-ssl

pwd

EXP_ROOT="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/wavlm/COCO_p2_large"

CFG_FILE="/work/vjsalt22/hsuanfu/audio-visual-ssl/config/KWClip/wavlm/COCO_p2_large.yaml"


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
    --gpus 4 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT
