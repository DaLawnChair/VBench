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

# video_paths=("/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sink_0_local_attn_21/sf_cf_extended")
# video_paths=("/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sink_0_local_attn_21/sf_cf_extended")


# video_paths=("/shared/john/AR_dawood/Causal-Forcing/vbench_long/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/"
# "/shared/john/AR_dawood/infinity-rope/vbench_long/self_forcing_dmd_plus_inf_rope/sf_cf_extended/vbench_scores/full_sf_all_dims/"
# "/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sf_cf_extended/vbench_scores/full_sf_all_dims/"
# "/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sink_1_local_attn_6/sf_cf_extended/vbench_scores/full_sf_all_dims"
# "/shared/john/AR_dawood/AR/vbench_long/rowwise_double_base_from_20k/iter_4000/sink_0_local_attn_21/sf_cf_extended/vbench_scores/full_sf_all_dims")
video_paths=("/shared/john/AR_dawood/Causal-Forcing/vbench_long/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/")
video_path=${video_paths[0]}
device="4"
echo $video_path

# dimensions=(background_consistency) #(background_consistency temporal_flickering)

# dimensions=(background_consistency subject_consistency aesthetic_quality \
#             imaging_quality object_class multiple_objects color)
# dimensions=(background_consistency subject_consistency aesthetic_quality \
#             imaging_quality object_class multiple_objects color \
#             spatial_relationship scene temporal_style overall_consistency \
#             human_action temporal_flickering motion_smoothness dynamic_degree \
#             appearance_style)

## do remaining dimensions 
# dimensions=(temporal_flickering_ratio unitemp_flickering_ratio)
dimensions=(temporal_flickering_ratio)

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"

    for dim in "${dimensions[@]}"; do 

        CUDA_VISIBLE_DEVICES="${device}" python3 vbench2_beta_long/eval_long.py \
            --videos_path "$video_path" \
            --mode long_custom_input_original \
            --output_path $video_path/vbench_scores/full_sf_all_dims/ \
            --dimension $dim \
            --dev_flag \
            --num_of_samples_per_prompt 1 # \
            # --static_filter_flag #[][] remove static filtering flag as it leads to 0 evalauted videos
    done
done

# dimensions=(unitemp_flickering_ratio)
# for video_path in "${video_paths[@]}"; do
#     echo "Evaluating videos in: $video_path"

#     for dim in "${dimensions[@]}"; do 

#         CUDA_VISIBLE_DEVICES="${device}" python3 vbench2_beta_long/eval_long.py \
#             --videos_path "$video_path" \
#             --mode long_custom_input_original \
#             --output_path $video_path/vbench_scores/full_sf_all_dims/ \
#             --dimension $dim \
#             --dev_flag \
#             --num_of_samples_per_prompt 1 # \
#             # --static_filter_flag #[][] remove static filtering flag as it leads to 0 evalauted videos
#     done
# done
date
