#!bin/bash
cd "/mnt/md0/user_jeff/audio-visual-ssl/zerospeech"
for idx in {0..14};
do
    echo "Evaluate layer ${idx}"
    conda activate avssl
    out_dir="/mnt/md0/user_jeff/zerospeech2021/embeddings/COCO_SpeechCLIP_p2/hidden_state_${idx}"

    python3 getEmbeddings.py \
        --inference_bsz 1 \
        --task_input_dir "/mnt/md0/dataset/zerospeech2021/phonetic" \
        --model_cls_name "KWClip_GeneralTransformer" \
        --model_ckpt "${cp}" \
        --feat_select_idx ${idx} \
        --task_name "phonetic" \
        --run_dev \
        --run_test \
        --output_result_dir $out_dir

    conda activate zerospeech2021
    DATASET="/mnt/md0/dataset/zerospeech2021/"
    RESULT="/mnt/md0/user_jeff/zerospeech2021/result/phonetic/COCO_SpeechCLIP_p2/hidden_state_${idx}"
    mkdir -p $RESULT
    # zerospeech2021-validate ${DATASET} ${out_dir} --no-phonetic --no-lexical --no-syntactic -j4 --only-dev
    zerospeech2021-evaluate --no-semantic --no-lexical --no-syntactic $DATASET $out_dir -j4 -o $RESULT
    # mv score_phonetic_test_correlation.csv $RESULT
    # mv score_phonetic_test_pairs.csv $RESULT
    rm -rf $out_dir

    echo "Done evaluation of layer ${idx}!"
done