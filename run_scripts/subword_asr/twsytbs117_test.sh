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

model_class="cascadedCLIP_local"
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/lr_1.0e-$i/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs/subword_asr300/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs/subword_asr1k/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs-plus/subword_asr1k"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs-plus/subword_asr1k/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/fast-vgs-plus/subword_asr300/1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/train_Flickr_kwCLIP_multGPU_integrated_large/lr_1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/KW_bsz256_WS_c1Mp1_flickr_c0.5p0.5/lr_1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/hubert/lr_1.0e-4/dev-best.ckpt"
# ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr1k/hubert/lr_1.0e-4/dev-best.ckpt"
ckpt_path="/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr300/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/lr_1.0e-4/dev-best.ckpt"
upstream_ckpt_path="/work/vjsalt22/atosystem/audio-visual-ssl/exp/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch_248-step_29132-val_recall_mean_1_11.1800.ckpt"

python3 run_downstream.py -m evaluate -e $ckpt_path \
        -o " \
        config.downstream_expert.corpus.path='/work/vjsalt22/dataset/LibriSpeech100',, \
        args.upstream_ckpt=${upstream_ckpt_path}
        "