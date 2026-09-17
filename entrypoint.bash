#!/bin/bash
# Runs the calibration against isaac_format/ (config.yaml + camera/ + lidar/
# + gt/), the Isaac Sim capture layout lcba.py's Isaac-format path expects.
# isaac_format/ is precomputed on the host (or regenerated on demand), not
# built by this entrypoint every run - see convert_input_to_isaac_format.py
# below, and the mounted isaac_format volume in compose.yaml. lcba.py itself
# auto-discovers apriltag ground truth from isaac_format/gt/apriltag_*.csv
# once path_data contains a config.yaml, and the reference map is still the
# surveyed FARO scan, not the isaac_format capture's own LiDAR scans.

# To (re)generate isaac_format/ from input/ + reference/apriltag_coords.csv:
# python3 calib/scripts/convert_input_to_isaac_format.py \
#     --path_data input \
#     --apriltag_file reference/apriltag_coords.csv \
#     --path_out isaac_format

python3 calib/scripts/lcba.py \
    --map-path reference/reference_pointcloud.ply \
    --path_data isaac_format \
    --path_out output \
    --experiment_name result1



