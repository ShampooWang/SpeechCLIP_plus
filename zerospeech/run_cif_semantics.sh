#!bin/bash
idx="cif"
echo "Evaluate layer ${idx}"
conda activate avssl
out_dir="/mnt/md0/user_jeff/zerospeech2021/embeddings/Flickr_c+_small_last/hidden_state_${idx}"

python3 getEmbeddings.py \
    --inference_bsz 1 \
    --model_cls_name "KWClip_GeneralTransformer_plus" \
    --model_ckpt "/mnt/md0/user_jeff/Checkpoints/Flickr_c+_base_last/epoch=248-step=116531-val_recall_mean_10=31.4100.ckpt" \
    --feat_select_idx ${idx} \
    --task_name "semantic" \
    --run_dev \
    --output_result_dir $out_dir

conda activate zerospeech2021
DATASET="/mnt/md0/dataset/zerospeech2021/"
RESULT="/mnt/md0/user_jeff/zerospeech2021/result/Flickr_c+_small_last/hidden_state_${idx}"
mkdir $RESULT
# zerospeech2021-validate ${DATASET} ${out_dir} --no-phonetic --no-lexical --no-syntactic -j4 --only-dev
zerospeech2021-evaluate --no-phonetic --no-lexical --no-syntactic $DATASET $out_dir -j4
mv score_semantic_dev_correlation.csv $RESULT
mv score_semantic_dev_pairs.csv $RESULT
rm -rf $out_dir

echo "Done evaluation of layer ${idx}!"
