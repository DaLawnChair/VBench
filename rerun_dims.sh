
#!/usr/bin/env bash
# Re-run one or more VBench dimensions for an existing result folder, then
# regenerate final_scores.txt. Use when run_sf_cf_eval_sleep.sh reported a
# failed dimension (it skips cal_final_from_folder.py in that case).
#
# Usage (from the VBench repo root, vbench env active):
#   DEVICE=5 bash rerun_dims.sh <video_path> [dimension ...]
#   dimensions default to: scene
# <video_path> is the folder containing videos_vbench/ and vbench_scores/.
set -uo pipefail

cd "$(dirname "$0")"

device="${DEVICE:?set DEVICE=<gpu index>}"
video_path="${1:?usage: DEVICE=N bash rerun_dims.sh <video_path> [dimension ...]}"
video_path="${video_path%/}"
shift
dims=("$@")
[[ ${#dims[@]} -eq 0 ]] && dims=(scene)

VIDEO_SUBDIR=videos_vbench
out="$video_path/vbench_scores/full_sf_all_dims"

n=$(ls "$video_path/$VIDEO_SUBDIR"/*.mp4 2>/dev/null | wc -l)
echo "videos: $n (expected 1046)"

failed=0
for dimension in "${dims[@]}"; do
    echo "=== $dimension ==="
    if ! CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
        --videos_path "$video_path/$VIDEO_SUBDIR" \
        --mode custom_input \
        --full_json_dir custom_small_vbench_set/sf_all_dimensions_extended_to_base_special_prompts.json \
        --prompt_file custom_small_vbench_set/sf_prompts_extended_to_base_special_prompts_mapping.json \
        --output_path "$out/" \
        --dimension "$dimension" 2>&1 | grep -v 'required video is not found'; then
        :
    fi
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "FAILED: $dimension" >&2
        failed=1
    fi
done

if [[ $failed -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=$device python3 scripts/cal_final_from_folder.py --folder "$out"
else
    echo "skipping cal_final_from_folder.py: a dimension failed" >&2
    exit 1
fi


# Some notes on it:
# - It filters out the flood of `-1…-4 not found` warnings, so a real traceback is visible this time.
# - Leftover orphan `*_full_info.json` files from the failed attempt shouldn't matter. `cal_final_from_folder.py` presumably reads the `*_eval_results.json` files, but check that in the main conversation before relying on it.
# - A related fix to consider: have `run_sf_cf_eval_sleep.sh` `tee` each dimension's output into a per-folder log. That way the reason for the next failure gets saved instead of lost with the terminal.