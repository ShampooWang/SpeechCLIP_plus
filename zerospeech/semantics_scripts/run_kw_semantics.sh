#!bin/bash
# /mnt/md0/user_jeff/Checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/Flickr_cascaded/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt
cd "/mnt/md0/user_jeff/audio-visual-ssl/zerospeech"
for idx in {13..14};
do
# idx=25
    echo "Evaluate layer ${idx}"
    conda activate speechclip+
    out_dir="/mnt/md0/user_jeff/zerospeech2021/embeddings/COCO_SpeechCLIP_c2_freeze/hidden_state_${idx}"

    python3 getEmbeddings.py \
        --inference_bsz 1 \
        --model_cls_name "SpeechCLIP" \
        --model_ckpt "/mnt/md0/user_jeff/Checkpoints/speechclip+/parallel/coco/COCO_SpeechCLIP_p2/epoch=13-step=129527-val_recall_mean_10=68.7405.ckpt" \
        --feat_select_idx ${idx} \
        --task_name "semantic" \
        --run_dev \
        --output_result_dir $out_dir

    conda activate zerospeech2021
    DATASET="/mnt/md0/dataset/zerospeech2021/"
    RESULT="/mnt/md0/user_jeff/zerospeech2021/result/COCO_SpeechCLIP_p2_reproduce/hidden_state_${idx}"
    mkdir -p $RESULT
    # zerospeech2021-validate ${DATASET} ${out_dir} --no-phonetic --no-lexical --no-syntactic -j4 --only-dev
    zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $out_dir -j4
    mv score_semantic_dev_correlation.csv $RESULT
    mv score_semantic_dev_pairs.csv $RESULT
    rm -rf $out_dir

    echo "Done evaluation of layer ${idx}!"
done