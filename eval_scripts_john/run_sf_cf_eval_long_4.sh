#!/bin/bash

export VBENCH_CACHE_DIR="/shared/john/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

export MASTER_PORT=29507
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/old_videos")
# video_paths=("/shared/john/AR_dawood/infinity-rope/vbench_long/self_forcing_dmd_plus_inf_rope/sf_cf_extended")
# video_paths=("/shared/john/AR_dawood/Causal-Forcing/videos/vbench_self_forcing_dmd")

video_paths=("/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sink_1_local_attn_6/sf_cf_extended")
video_path=${video_paths[0]}
device="7"
echo $video_path

# dimensions=(background_consistency) #(background_consistency temporal_flickering)

dimensions=(background_consistency subject_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color)
# dimensions=(subject_consistency background_consistency aesthetic_quality \
#             imaging_quality object_class multiple_objects color \
#             spatial_relationship scene temporal_style overall_consistency \
#             human_action temporal_flickering motion_smoothness dynamic_degree \
#             appearance_style)

## presave videos since the av package is not avaialble yet
# conda activate vbench_john
# CUDA_VISIBLE_DEVICES="${device}" python3 vbench2_beta_long/eval_long.py \
#     --videos_path "$video_path" \
#     --mode long_custom_input \
#     --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
#     --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
#     --output_path $video_path/vbench_scores/full_sf_all_dims \
#     --dimension subject_consistency \
#     --dev_flag \
#     --static_filter_flag \
#     --num_of_samples_per_prompt 1 \
#     --bg_clip2clip_feat_extractor clip # note: default is dreamsim, but packages are too different to use

# conda activate vbench
# rm $video_path/vbench_scores/full_sf_all_dims/*

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"

    for dim in "${dimensions[@]}"; do 
        conda activate vbench
        if [ "$dim" == "background_consistency" ]; then
            conda activate vbench_john # need to change environments for dreamsim package, too many incompatible with vbench
            CUDA_VISIBLE_DEVICES="${device}" python3 vbench2_beta_long/eval_long.py \
                --videos_path "$video_path" \
                --mode long_custom_input \
                --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
                --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
                --output_path $video_path/vbench_scores/full_sf_all_dims/ \
                --dimension $dim \
                --dev_flag \
                --num_of_samples_per_prompt 1 # \
                # --static_filter_flag #[][] remove static filtering flag as it leads to 0 evalauted videos
        else
            CUDA_VISIBLE_DEVICES="${device}" python3 vbench2_beta_long/eval_long.py \
                --videos_path "$video_path" \
                --mode long_custom_input \
                --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
                --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
                --output_path $video_path/vbench_scores/full_sf_all_dims/ \
                --dimension $dim \
                --dev_flag \
                --num_of_samples_per_prompt 1 # \
                # --static_filter_flag #[][] remove static filtering flag as it leads to 0 evalauted videos
        fi
    done
done

date
