#!bin/bash
cd /work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl
DATASET=/work/vjsalt22/dataset/LibriSpeech100
python3 preprocess/generate_len_for_bucket.py -i ${DATASET} -o data/librispeech -a .flac --n_jobs 12