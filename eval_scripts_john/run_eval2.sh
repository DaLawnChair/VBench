#!/bin/bash

export VBENCH_CACHE_DIR="/shared/john/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

export MASTER_PORT=29504
video_paths=("/shared/john/AR_dawood/Causal-Forcing/videos/vbench_causal_forcing_dmd")
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/videos/vbench_self_forcing_dmd")

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"

    CUDA_VISIBLE_DEVICES="5" python3 evaluate.py \
        --videos_path "$video_path" \
        --mode vbench_standard \
        --output_path $video_path/vbench_scores/full \
        --dimension subject_consistency background_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color \
            spatial_relationship scene temporal_style overall_consistency \
            human_action temporal_flickering motion_smoothness dynamic_degree \
            appearance_style    
done
