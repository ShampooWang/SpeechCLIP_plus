#!/bin/bash
#SBATCH --job-name=w2vuPool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="ENT211069"
#SBATCH --partition=gp4d


lscpu | egrep 'CPU\(s\)'
cd /work/twsezjg982/atosytem/audio-visual-ssl

pwd

cd /work/twsezjg982/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl

srun python3 run_downstream.py -n cascadedCLIP_local_fixedZEROCLS_test -m train -u cascadedCLIP_local  -d fluent_commands -k /work/twsezjg982/hsuanfu/audio-visual-ssl/exp/sc_c/kw/epoch=89-step=28170-val_recall_mean_1=24.7300.ckpt

