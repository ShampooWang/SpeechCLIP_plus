#!bin/bash
python3 km_label.py \
        --name "wavlm_base_plus" \
        --feat_select_idx "hidden_state_11" \
        --input_file_list "/mnt/md0/dataset/flickr/audio_path_train.txt" \
        --bsz 32 \
        --output_file "cluster_id.json" \