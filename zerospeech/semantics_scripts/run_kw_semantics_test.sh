#!bin/bash
for idx in {13..14};
do
# idx=15
    # echo "Evaluate layer ${idx}"
    # conda activate avssl
    # out_dir="/mnt/md0/user_jeff/zerospeech2021/test/COCO_SpeechCLIP_p2/hidden_state_${idx}"

    # python3 getEmbeddings.py \
    #     --inference_bsz 1 \
    #     --model_cls_name "KWClip_GeneralTransformer" \
    #     --model_ckpt "${cp}" \
    #     --feat_select_idx ${idx} \
    #     --task_name "semantic" \
    #     --run_test \
    #     --output_result_dir $out_dir

    conda activate zerospeech2021
    DATASET="/mnt/md0/dataset/zerospeech2021/"
    RESULT="/mnt/md0/user_jeff/zerospeech2021/test/COCO_SpeechCLIP_p2/hidden_state_${idx}"
    # mkdir -p $RESULT
    # zerospeech2021-validate ${DATASET} ${out_dir} --no-phonetic --no-lexical --no-syntactic -j4 --only-dev
    zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $RESULT -j4
    mv score_semantic_dev_correlation.csv $RESULT
    mv score_semantic_dev_pairs.csv $RESULT

    echo "Done evaluation of layer ${idx}!"
done