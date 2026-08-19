#!/bin/bash

export VBENCH_CACHE_DIR="/shared/john/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

# python3
#VIDEOS_PATH="/shared/john/VBench/video"  
VIDEOS_PATH="/shared/john/sample_videos/causal-forcing"
#cd /shared/bofeng/code/VBench

# python3 evaluate.py \
#     --videos_path "$VIDEOS_PATH" \
#     --dimension subject_consistency background_consistency temporal_flickering \
#                 motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
#                 human_action temporal_style overall_consistency \
#     --mode custom_input

# video_paths=("/shared/john/AR_dawood/AR/vbench/mini/double_refine_gan/iter_7000/" 
# "/shared/john/AR_dawood/AR/vbench/mini/rowwise_double_base_from_20k/iter_4000")

video_paths=("/shared/john/AR_dawood/Causal-Forcing/videos/vbench_causal_forcing_dmd" "/shared/john/AR_dawood/Causal-Forcing/videos/vbench_self_forcing_dmd")


export MASTER_PORT=29504

for video_path in "${video_paths[@]}"; do
    echo "Evaluating videos in: $video_path"

    CUDA_VISIBLE_DEVICES="5" python3 evaluate.py \
        --videos_path "$video_path" \
        --mode vbench_standard \
        --full_json_dir  ./custom_small_vbench_set/VBench_subset_balanced_10_each.json \
        --output_path $video_path/vbench_scores/mini/ \
        --dimension subject_consistency background_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color \
            spatial_relationship scene temporal_style overall_consistency \
            human_action temporal_flickering motion_smoothness dynamic_degree \
            appearance_style    

        # CUDA_VISIBLE_DEVICES="5" python3 evaluate.py \
        # --videos_path "$video_path" \
        # --mode vbench_standard \
        # --full_json_dir  ./custom_small_vbench_set/VBench_subset_balanced_10_each.json \
        # --output_path $video_path/vbench_scores/ \
        # --dimension human_action temporal_flickering motion_smoothness dynamic_degree appearance_style
        # --dimension spatial_relationship scene temporal_style overall_consistency
        # --dimension imaging_quality object_class multiple_objects color
        # --dimension subject_consistency background_consistency aesthetic_quality # GOOD
done


# python3 scripts/cal_final_no_zip_score.py --file /shared/john/AR_dawood/AR/vbench/mini/double_refine_gan/iter_7000/vbench_scores/results_2026-07-03-19:22:57_eval_results.json
# python3 scripts/cal_final_no_zip_score.py --file /shared/john/AR_dawood/AR/vbench/mini/rowwise_double_base_from_20k/iter_4000/vbench_scores/results_2026-07-03-19:36:38_eval_results.json
# python3 scripts/cal_final_no_zip_score.py --file /shared/john/AR_dawood/Causal-Forcing/videos/vbench_causal_forcing_dmd/vbench_scores/results_2026-07-03-19:58:54_eval_results.json

# python3 scripts/cal_final_no_zip_score.py --file /shared/john/AR_dawood/Causal-Forcing/videos/vbench_self_forcing_dmd/vbench_scores/results_2026-07-03-20:10:00_eval_results.json