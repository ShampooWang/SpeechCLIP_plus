#!bin/bash
# The default dist_fn if not specified is "cosine_exp"
# as it yields the best result for almost all upstream
# Supported dist_fn: cosine, cityblock, euclidean, cosine_exp
dist_fn=cosine
cd $s3prl
# ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_c+_small_test_150k_1e-4_pad/epoch=314-step=147419-val_recall_mean_10=30.4300.ckpt"
ckpt_path="/mnt/md0/user_jeff/Checkpoints/Flickr_h+_small/epoch=48-step=22931-val_recall_mean_10=25.8800.ckpt"
exp_name="Flickr_h+_small_all"

for layer in {0..15};
do
# i=13
# layer=$i
echo "layer: $layer"

# dev
python3 run_downstream.py -m evaluate -t "dev" -u cascadedCLIP_local -l ${layer} \
    -d quesst14_dtw -n ExpName_${layer}_dev \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn \
    --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/Qbe/dev/$exp_name/$layer" \
    -k $ckpt_path

# test
python3 run_downstream.py -m evaluate -t "test" -u cascadedCLIP_local -l ${layer} \
    -d quesst14_dtw -n ExpName_${layer}_test \
    -o config.downstream_expert.dtwrc.dist_method=$dist_fn \
    --expdir "/mnt/md0/user_jeff/SUPERB/wavlm/Qbe/test/$exp_name/$layer" \
    -k $ckpt_path
done