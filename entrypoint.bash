#!/bin/bash
# Do the calibration
# This is just a call to a python script:
current_time=$(date +"%Y%m%d_%H%M")

python3 calib/scripts/lcba.py \
        --faro_file reference/reference_pointcloud.ply \
        --apriltag_file reference/apriltag_coords.txt \
        --path_data input \
        --path_out output \
        --tag-family tag16h5 \
        --experiment_name ${current_time} \
        $([ -f gt.yaml ] && echo --gt_file gt.yaml) | tee output/${current_time}.log



