#!bin/bash
cd "/mnt/md0/user_jeff/audio-visual-ssl/zerospeech"
# export ZEROSPEECH2021_TEST_GOLD=true

for idx in {0..14};
do
# idx=15
    echo "Evaluate layer ${idx}"
    conda activate speechclip+
    out_dir="/mnt/md0/user_jeff/zerospeech2021/embeddings/flickr_c+/hidden_state_${idx}"

    python3 getEmbeddings.py \
        --inference_bsz 1 \
        --model_cls_name "SpeechCLIP_plus" \
        --model_ckpt "/mnt/md0/user_jeff/Checkpoints/speechclip+/cascaded/flickr/flickr_c+/epoch=105-step=49608-val_recall_mean_10=33.3000.ckpt" \
        --feat_select_idx ${idx} \
        --task_name "semantic" \
        --run_dev \
        --run_test \
        --output_result_dir $out_dir

    conda activate zerospeech2021
    DATASET="/mnt/md0/dataset/zerospeech2021/"
    RESULT="/mnt/md0/user_jeff/zerospeech2021/result/flickr_c+/hidden_state_${idx}"
    mkdir -p $RESULT
    # zerospeech2021-validate ${DATASET} ${out_dir} --no-phonetic --no-lexical --no-syntactic -j4 --only-dev
    zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $out_dir -j4
    mv score_semantic_dev_correlation.csv $RESULT
    mv score_semantic_dev_pairs.csv $RESULT
    mv score_semantic_test_correlation.csv $RESULT
    mv score_semantic_test_pairs.csv $RESULT
    rm -rf $out_dir

    echo "Done evaluation of layer ${idx}!"
done