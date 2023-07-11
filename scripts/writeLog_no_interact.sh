#!bin/bash
LogsRoot="/mnt/md0/user_jeff/logs"
bash -i ${1} -type -f 2>&1 | tee "${LogsRoot}/${2}"