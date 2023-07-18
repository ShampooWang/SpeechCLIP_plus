lscpu | egrep 'CPU\(s\)'
cd /mnt/md0/user_jeff/audio-visual-ssl

EXP_ROOT="/mnt/md0/user_jeff/Checkpoints/speechclip+/hybrid/flickr/flickr_h+"
CFG_FILE="/mnt/md0/user_jeff/audio-visual-ssl/config/speechclip+/hybrid/flickr_h2.yaml"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --config $CFG_FILE \
    --gpus 3 \
    --njobs 12 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT