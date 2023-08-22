#!/bin/bash
#SBATCH --job-name=devTest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/SpeechCLIP_plus/logs/speechclip+/cos_semantics/devTest.log

cd /work/jgtf0322/SpeechCLIP_plus

python compute_cos_alignments.py "/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/cascaded/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt" \
                    "/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/cascaded/test/retokenizeText/keywords_ep0.json" \
                    "/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/cascaded/test/retokenizeText/cos_ali.json" \