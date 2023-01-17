#!bin/bash
python3 compute_kmeans.py \
        --name "wavlm_base_plus" \
        --input_file_list "/mnt/md0/dataset/flickr/audio_path_train.txt" \
        --bsz 32 \
        --output_file "cluster_id.json" \