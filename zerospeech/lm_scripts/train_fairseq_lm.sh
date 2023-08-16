#!bin/bash

fairseq-train --fp16 /mnt/md1/user_jeffwang/Dataset/librispeech/fairseq-bin-data \
    --task masked_lm --criterion masked_lm \
    --save-dir /mnt/md1/user_jeffwang/zerospeech2021/checkpoints/BERT/coco_p+_kssplit \
    --keep-last-epochs 1 \
    --train-subset train \
    --num-workers 4 \
    --arch roberta_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --mask-multiple-length 10 --mask-prob 0.5 --mask-stdev 10 \
    --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
    --max-tokens 4096 --update-freq 4 --max-update 250000 \
    --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test