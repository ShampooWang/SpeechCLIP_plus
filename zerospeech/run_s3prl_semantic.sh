#!bin/bash
for idx in {0..12};
do
    conda activate avssl
    out_dir="/mnt/md0/user_jeff/zerospeech2021/embeddings/wavlm/hidden_state_${idx}"

    python3 getEmbeddings.py \
        --s3prl true \
        --inference_bsz 1 \
        --model_cls_name "wavlm_base_plus" \
        --feat_select_idx "hidden_state_${idx}" \
        --task_name "semantic" \
        --run_dev \
        --output_result_dir $out_dir

    conda activate zerospeech2021
    DATASET="/mnt/md0/dataset/zerospeech2021/"
    RESULT="/mnt/md0/user_jeff/zerospeech2021/result/wavlm/hidden_state_${idx}"
    mkdir $RESULT
    zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $out_dir
    mv score_semantic_dev_correlation.csv $RESULT
    mv score_semantic_dev_pairs.csv $RESULT
    rm -rf $out_dir
    
    echo "Evaluation done!"
done