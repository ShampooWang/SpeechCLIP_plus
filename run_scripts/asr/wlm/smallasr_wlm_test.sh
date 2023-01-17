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
# sudo apt-get update
# sudo apt-get install libboost-system-dev libboost-filesystem-dev libboost-chrono-dev libboost-program-options-dev libboost-test-dev libboost-thread-dev libboost-iostreams-dev libbz2-dev bzip2 fftw3 fftw3-dev pkg-config ocl-icd-opencl-dev

model_list=("cascaded" "integrated")
model_ckpt_list=( \
'/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_multiAtt_c_flickr/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt' \
'/work/vjsalt22/atosystem/audio-visual-ssl/exp/old/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/epoch_248-step_29132-val_recall_mean_1_11.1800.ckpt' \
)
# exp_name="1e-4_kw_lr_1e-3_heads_1_keyword_8"

model_class="cascadedCLIP_local"
# ckpt_path="/work/twsytbs117/new-audio-visual-ssl/tmp_tools/s3prl/s3prl/result/downstream/subword_asr/KW_bsz256_WS_integratedM_flickr_c0.5p0.5/lr_1.0e-1/states-200000.ckpt"
# python3 run_downstream.py -m evaluate -t "test-clean" -e $ckpt_path
for i in "${!model_list[@]}"
do
    echo ${model_list[i]}
    # echo ${model_ckpt_list[i]}
    ckpt_path="/work/vjsalt22/hsuanfu/superb_result/small_model/asr/${model_list[i]}/1.0e-4/dev-clean-best.ckpt"
    python3 run_downstream.py -m evaluate -t "test-clean" -e $ckpt_path \
            -o "\
            config.downstream_expert.datarc.decoder_args.decoder_type='kenlm',, \
            config.downstream_expert.datarc.decoder_args.kenlm_model='/work/vjsalt22/hsuanfu/audio-visual-ssl/4-gram_LM/4-gram.arpa.gz',, \
            config.downstream_expert.datarc.decoder_args.lexicon='/work/vjsalt22/hsuanfu/audio-visual-ssl/4-gram_LM/librispeech_lexicon.lst',, \
            config.downstream_expert.datarc.libri_root='/work/vjsalt22/dataset/LibriSpeech100',, \
            args.upstream_ckpt=${model_ckpt_list[i]}
            "
done