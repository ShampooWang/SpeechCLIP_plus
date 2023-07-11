#!bin/bash
cd /mnt/md0/dataset/quesst14Database/scoring
RESULT_DIR="/mnt/md0/user_jeff/SUPERB/wavlm/Qbe"
EXP_NAME="Flickr_h+_small_all"

for layer in {14..15}
do
# dev
./score-TWV-Cnxe.sh ${RESULT_DIR}/dev/${EXP_NAME}/${layer} \
    groundtruth_quesst14_dev -10

# test
./score-TWV-Cnxe.sh ${RESULT_DIR}/test/${EXP_NAME}/${layer} \
    groundtruth_quesst14_eval -10
done