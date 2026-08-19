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

# /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b/iter_5000/sf_cf_extended/
# video_paths=(
#     /shared/john/AR_dawood/AR/vbench/260724_bidirectional_all_step_all_noise_mean_score/iter_3500/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b_renoise_250_to_0/iter_2000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b_grad_on_all_steps/iter_3500/sf_cf_extended/

#     /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b/iter_11000/sf_cf_extended
#     /shared/john/AR_dawood/AR/vbench/rcm/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_2500/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_5000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_7000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_last_step_dmd/iter_7000/sf_cf_extended/
# )

# video_paths=(
#     /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_10000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_37000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_24000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_cache_bidir_first_critic_match/iter_25000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_cache_refine_dual_dmd_base_14b/iter_9000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_gan_dmd_1_step_14b/iter_1000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/double_gan_dmd_1_step_14b_37k/iter_1500/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_1_step_dmd_normal_noise/iter_6000/sf_cf_extended/
#     /shared/john/AR_dawood/AR/vbench/bidir_1_step_dmd_high_noise/iter_6000/sf_cf_extended/
# )

video_paths=(
    /shared/john/AR_dawood/AR/vbench/260720_bidirectional_long_train_real_score_14b/iter_6000/sf_cf_extended/
)

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"
    # for dimension in "${dimensions[@]}"; do

        # CUDA_VISIBLE_DEVICES="6" python3 evaluate.py \
        #     --videos_path "$video_path" \
        #     --mode custom_input \
        #     --full_json_dir /shared/john/VBench/custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
        #     --prompt_file /shared/john/VBench/custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
        #     --output_path $video_path/vbench_scores/full_sf_all_dims/ \
        #     --dimension $dimension

    # CUDA_VISIBLE_DEVICES="4" python3 evaluate.py \
    #     --videos_path "$video_path" \
    #     --mode custom_input_original \
    #     --output_path $video_path/vbench_scores/full_sf_all_dims \
    #     --dimension temporal_flickering_ratio unitemp_flickering_ratio

    CUDA_VISIBLE_DEVICES="7" python3 evaluate.py \
        --videos_path "$video_path" \
        --mode custom_input \
        --full_json_dir /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
        --prompt_file /shared/john/VBench/custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
        --output_path $video_path/vbench_scores/cf_dynamic_degree \
        --dimension dynamic_degree
    done
done

date
