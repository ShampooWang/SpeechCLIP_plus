lscpu | egrep 'CPU\(s\)'
cd /mnt/md1/user_jeffwang/audio-visual-ssl

EXP_ROOT="/mnt/md1/user_jeffwang/Checkpoints/speechclip+/hybrid/coco/coco_h_large"
CFG_FILE="/mnt/md1/user_jeffwang/audio-visual-ssl/config/speechclip+/hybrid/coco/coco_h2_large.yaml"
mkdir $EXP_ROOT

python3 run_task.py \
    "TrainSpeechCLIP_plus" \
    --config $CFG_FILE \
    --gpus 2 \
    --njobs 8 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT