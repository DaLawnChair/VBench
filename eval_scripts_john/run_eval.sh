#!/bin/bash

export VBENCH_CACHE_DIR=/shared/bofeng/code/VBench/pretrained

VIDEOS_PATH="/shared/bofeng/code/video"  

cd /shared/bofeng/code/VBench

python3 evaluate.py \
    --videos_path "$VIDEOS_PATH" \
    --dimension subject_consistency background_consistency temporal_flickering \
                motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
                human_action temporal_style overall_consistency \
    --mode custom_input
