#!/bin/bash

fairseq-preprocess --only-source \
    --trainpref /mnt/md1/user_jeffwang/Dataset/librispeech/quantized/train/fairseq_train.txt \
    --validpref /mnt/md1/user_jeffwang/Dataset/librispeech/quantized/dev-clean/fairseq_dev_clean.txt \
    --testpref /mnt/md1/user_jeffwang/Dataset/librispeech/quantized/test-clean/fairseq_test_clean.txt \
    --destdir /mnt/md1/user_jeffwang/Dataset/librispeech/fairseq-bin-data \
    --workers 20