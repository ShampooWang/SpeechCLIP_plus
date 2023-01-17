#!/bin/bash
## Usage: run_semantic.sh DATASET OUTPUT_DIR KIND CORPUS
##
## Generates the semantic submission for BERT and LSTM baselines.
## Writes OUTPUT_DIR/{bert,lstm}/semantic/CORPUS/*.txt
##
## Parameters:
##  DATASET     path to the zerospeech2021 dataset
##  OUTPUT_DIR  directory to write results on
##  KIND        must be 'dev' or 'test'
##  CORPUS      must be 'synthetic' or 'librispeech'
FEATURES_PY=/work/vjsalt22/hsuanfu/audio-visual-ssl/zerospeech/build_hubert_features.py
# MODEL_CKPT="/work/vjsalt22/hsuanfu/audio-visual-ssl/exp/sc_c/epoch=29-step=9390-val_recall_mean_1=30.7600.ckpt"

for e in "dev" "test"
do 
        for d in "librispeech" "synthetic"
        do 
                OUTPUT_DIR="result/hubert_hid8/semantic/${e}/${d}"
                python $FEATURES_PY \
                        "/work/vjsalt22/dataset/zerospeech2021/semantic/${e}_${d}.txt" \
                        $OUTPUT_DIR
        done
done
echo "finished at $(date)"
exit 0