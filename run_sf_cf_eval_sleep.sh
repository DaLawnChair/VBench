#!/bin/bash
# VBench scoring driver (John's patched fork). Must be run from this VBench
# checkout (evaluate.py / custom_small_vbench_set/ are referenced relatively).
#
# usage:
#   bash run_sf_cf_eval_sleep.sh                       # scores the hardcoded video_paths below
#   bash run_sf_cf_eval_sleep.sh <video_path> [...]    # scores the given paths instead
#
# Each <video_path> is an iteration dir that contains (or will contain)
# $VIDEO_SUBDIR/*.mp4 (1046 files: 946 sf prompts + 100 cf dynamic-degree
# prompts, one "<truncated prompt>-0.mp4" each). Scores are written to
# <video_path>/vbench_scores/. The GPU can be overridden with the VBENCH_DEVICE
# env var (default: the hardcoded `device` below).

export VBENCH_CACHE_DIR="/mnt/ddn/george/VBench/pretrained"

export TORCH_HOME=$VBENCH_CACHE_DIR
export HF_HOME=$VBENCH_CACHE_DIR
export VBENCH_CACHE_DIR=$TORCH_HOME
export VBENCH2_CACHE_DIR=$TORCH_HOME

# respect a caller-exported MASTER_PORT so concurrent drivers don't collide
export MASTER_PORT="${MASTER_PORT:-29518}"

device="${VBENCH_DEVICE:-2}"

echo "device: $device"

video_paths=(
    # /shared/john/AR_dawood/AR/vbench/double_cache_bidir_first_critic_match/iter_10000/bidirectional/sf_cf_extended/
    # /mnt/ddn/george/diffusion/john-AR/vbench_results/260810_dc_dataforcing_high_renoise/iter_6k/double_cache/
    "/shared/john/AR_dawood/AR/vbench/double_cache_14b_msm_continued_11k/bidirectional_redo/iter_11500/sf_cf_extended"
)
# command-line arguments override the hardcoded list above
if [[ $# -gt 0 ]]; then
    video_paths=("$@")
fi

# subdir of each video_path that holds the mp4s (matches --out-dir in
# AR/inference_double_cache_vbench.sh)
VIDEO_SUBDIR="videos_vbench"
EXPECTED_VIDEO_COUNT=1046
CHECK_INTERVAL="10m"

dimensions=(subject_consistency background_consistency aesthetic_quality \
            imaging_quality object_class multiple_objects color \
            spatial_relationship scene temporal_style overall_consistency \
            human_action temporal_flickering motion_smoothness dynamic_degree \
            appearance_style)
# dimensions=(scene)

wait_for_videos() {
    local video_dir="$1"
    local video_count

    while true; do
        if [[ ! -d "$video_dir" ]]; then
            echo "$(date): Folder does not exist: $video_dir"
            echo "Checking again in $CHECK_INTERVAL..."
            sleep "$CHECK_INTERVAL"
            continue
        fi

        video_count=$(
            find "$video_dir" \
                -maxdepth 1 \
                -type f \
                -iname '*.mp4' \
                -printf '.' |
            wc -c
        )

        echo "$(date): Found $video_count / $EXPECTED_VIDEO_COUNT MP4 files in:"
        echo "  $video_dir"

        if [[ "$video_count" -eq "$EXPECTED_VIDEO_COUNT" ]]; then
            echo "Expected video count reached. Starting evaluation."
            break
        fi

        echo "Video count is not exactly $EXPECTED_VIDEO_COUNT."
        echo "Checking again in $CHECK_INTERVAL..."
        sleep "$CHECK_INTERVAL"
    done
}

echo "Scoring ${#video_paths[@]} video path(s):"
printf '  %s\n' "${video_paths[@]}"

# every failed step, as "<video_path>: <step>"; reported at the end
failures=()

fail() {
    failures+=("$1: $2")
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "ERROR: $2 FAILED for $1" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
}

for video_path in "${video_paths[@]}"; do
    video_path="${video_path%/}"
    wait_for_videos "$video_path/$VIDEO_SUBDIR"

    echo "Evaluating videos in: $video_path/$VIDEO_SUBDIR"
    dims_failed=0
    for dimension in "${dimensions[@]}"; do

        if ! CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
            --videos_path "$video_path/$VIDEO_SUBDIR" \
            --mode custom_input \
            --full_json_dir custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
            --prompt_file custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
            --output_path "$video_path/vbench_scores/full_sf_all_dims/" \
            --dimension $dimension; then
            fail "$video_path" "evaluate.py --dimension $dimension"
            dims_failed=1
        fi
    done

    # a failed dimension would make the summary wrong, so skip it
    if [[ $dims_failed -eq 0 ]]; then
        echo "scripts/cal_final_from_folder.py for folder $video_path/vbench_scores/full_sf_all_dims"
        if ! CUDA_VISIBLE_DEVICES=$device python3 scripts/cal_final_from_folder.py --folder "$video_path/vbench_scores/full_sf_all_dims"; then
            fail "$video_path" "cal_final_from_folder.py"
        fi
    else
        fail "$video_path" "cal_final_from_folder.py (skipped: a dimension failed)"
    fi

    if ! CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
        --videos_path "$video_path/$VIDEO_SUBDIR" \
        --mode custom_input \
        --full_json_dir custom_small_vbench_set/cf_prompts_extended_dynamic_degree.json \
        --prompt_file custom_small_vbench_set/cf_prompts_extended_dynamic_degree_mapping.json \
        --output_path "$video_path/vbench_scores/cf_dynamic_degree" \
        --dimension dynamic_degree; then
        fail "$video_path" "evaluate.py cf dynamic_degree"
    fi
done


date

if [[ ${#failures[@]} -gt 0 ]]; then
    echo "================================================================" >&2
    echo "VBench scoring finished with ${#failures[@]} FAILED step(s):" >&2
    printf '  %s\n' "${failures[@]}" >&2
    echo "================================================================" >&2
    exit 1
fi
echo "VBench scoring finished: all steps succeeded."
