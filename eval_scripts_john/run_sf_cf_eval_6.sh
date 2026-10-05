#!/bin/bash

export VBENCH_CACHE_DIR="/shared/john/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

export MASTER_PORT=29513
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/old_videos")
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/")
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/videos/vbench_self_forcing_dmd")
# video_paths=("/shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_37000/sf_cf_extended/")
video_paths=("/shared/john/AR_dawood/AR/vbench/double_cache_refine_dual_dmd_base_14b/iter_9000/sf_cf_extended")

# --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
# --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"

    CUDA_VISIBLE_DEVICES="7" python3 evaluate.py \
        --videos_path "$video_path" \
        --mode custom_input \
        --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
        --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
        --output_path $video_path/vbench_scores/full_sf_all_dims \
        --dimension subject_consistency background_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color \
            spatial_relationship scene temporal_style overall_consistency \
            human_action temporal_flickering motion_smoothness dynamic_degree \
            appearance_style   

    # CUDA_VISIBLE_DEVICES="5" python3 evaluate.py \
    #     --videos_path "$video_path" \
    #     --mode custom_input \
    #     --full_json_dir /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
    #     --prompt_file /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
    #     --output_path $video_path/vbench_scores/cf_dynamic_degree \
    #     --dimension dynamic_degree
done

date