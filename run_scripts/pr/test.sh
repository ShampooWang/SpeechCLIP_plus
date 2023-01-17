#!/bin/bash
#SBATCH --job-name=w2vuPool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST111160"
#SBATCH --partition=gp4d


lscpu | egrep 'CPU\(s\)'

cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl

# exp_name="1e-4_kw_lr_1e-3_heads_1_keyword_8"

model_class="cascadedCLIP_local"
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/asr/train_Flickr_kwCLIP_multGPU_integrated_large/1.0e-4/dev-clean-best.ckpt"
ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/PR/flickr_p_small_MHA/1.0e-4/dev-best.ckpt"
python3 run_downstream.py -m evaluate -e $ckpt_path