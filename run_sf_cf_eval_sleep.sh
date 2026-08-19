#!/bin/bash

export VBENCH_CACHE_DIR="/shared/john/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

export MASTER_PORT=29518
# video_paths=(
#     "/shared/john/AR_dawood/AR/vbench/bidir_1_step_dmd_high_noise/iter_6000/sf_cf_extended/"
#     )

# video_paths=(
#     /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b/iter_11000/sf_cf_extended
#     /shared/john/AR_dawood/AR/vbench/rcm/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_2500/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_5000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_7000/sf_cf_extended/
# )

# video_paths=(
#     /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b/iter_6000/sf_cf_extended/
# )

video_paths=(
    # /shared/john/AR_dawood/AR/vbench/double_cache_bidir_first_critic_match/iter_10000/bidirectional/sf_cf_extended/
    # /shared/john/AR_dawood/AR/vbench/double_cache_bidir_first_critic_match/iter_10000/double_cache/sf_cf_extended/
    # /shared/john/AR_dawood/AR/vbench/double_cache_14b_msm_continued_11k/iter_11500/sf_cf_extended/
    # /shared/john/AR_dawood/AR/vbench/double_cache_14b_msm_continued_11k/iter_11500/sf_cf_extended
    # /shared/john/AR_dawood/AR/vbench/260730_dc_gan_dmd_all_steps_dmd_msm_ckpt/iter_1500/sf_cf_extended/
    /mnt/ddn/john/AR_dawood/AR/vbench/260811_dc_gan_dmd_all_steps_dmd_msm_ckpt_baseline_refine_t750_gpu8/iter_900/sf_cf_extended
)
EXPECTED_VIDEO_COUNT=1046
CHECK_INTERVAL="10m"


dimensions=(subject_consistency background_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color \
            spatial_relationship scene temporal_style overall_consistency \
            human_action temporal_flickering motion_smoothness dynamic_degree \
            appearance_style)


wait_for_videos() {
    local video_path="$1"
    local video_count

    while true; do
        if [[ ! -d "$video_path" ]]; then
            echo "$(date): Folder does not exist: $video_path"
            echo "Checking again in 30 minutes..."
            sleep "$CHECK_INTERVAL"
            continue
        fi

        video_count=$(
            find "$video_path" \
                -maxdepth 1 \
                -type f \
                -iname '*.mp4' \
                -printf '.' |
            wc -c
        )

        echo "$(date): Found $video_count / $EXPECTED_VIDEO_COUNT MP4 files in:"
        echo "  $video_path"

        if [[ "$video_count" -eq "$EXPECTED_VIDEO_COUNT" ]]; then
            echo "Expected video count reached. Starting evaluation."
            break
        fi

        echo "Video count is not exactly $EXPECTED_VIDEO_COUNT."
        echo "Checking again in 30 minutes..."
        sleep "$CHECK_INTERVAL"
    done
}


for video_path in "${video_paths[@]}"; do
    wait_for_videos "$video_path"

    echo "Evaluating videos in: $video_path"
    for dimension in "${dimensions[@]}"; do

        CUDA_VISIBLE_DEVICES="0" python3 evaluate.py \
            --videos_path "$video_path" \
            --mode custom_input \
            --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
            --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
            --output_path $video_path/vbench_scores/full_sf_all_dims/ \
            --dimension $dimension
    done
    CUDA_VISIBLE_DEVICES="0" python3 evaluate.py \
        --videos_path "$video_path" \
        --mode custom_input \
        --full_json_dir /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
        --prompt_file /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
        --output_path $video_path/vbench_scores/cf_dynamic_degree \
        --dimension dynamic_degree
done

    # CUDA_VISIBLE_DEVICES="4" python3 evaluate.py \
    #     --videos_path "$video_path" \
    #     --mode custom_input_original \
    #     --output_path $video_path/vbench_scores/full_sf_all_dims \
    #     --dimension temporal_flickering_ratio unitemp_flickering_ratio

    # CUDA_VISIBLE_DEVICES="5" python3 evaluate.py \
    #     --videos_path "$video_path" \
    #     --mode custom_input \
    #     --full_json_dir /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
    #     --prompt_file /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
    #     --output_path $video_path/vbench_scores/cf_dynamic_degree \
    #     --dimension dynamic_degree
date
