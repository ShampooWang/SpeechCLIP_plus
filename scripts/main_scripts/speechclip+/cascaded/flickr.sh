lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

EXP_ROOT="/mnt/md0/user_jeff/Checkpoints/speechclip+/cascaded/flickr/flickr_c+_cos_ent_pen"
CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/speechclip+/cascaded/flickr/flickr_c2.yaml"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --config $CFG_FILE \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT