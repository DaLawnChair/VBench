#!/usr/bin/env bash
# Visualise per-pixel RGB Euclidean distance between consecutive frames (source frame left, distance map right).
# Writes <videos_path>/rgbdiff_viz/<video>_rgbdiff.mp4 + .csv  (see scripts/plot_rgb_diff.py --help). CPU only.
# Usage: bash run_plot_rgb_diff.sh [videos_path]   (defaults to the path below)
set -euo pipefail
cd "$(dirname "$0")"

python=/shared/miniconda3/envs/vbench/bin/python

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k/videos_rsample"
CUDA_VISIBLE_DEVICES="" $python scripts/plot_rgb_diff.py \
    --videos_path "$videos_path"
    # --no_source           # distance map only, no source frame on the left
    # --out_dir <dir>       # default: $videos_path/rgbdiff_viz

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k_bidirectional/videos_rsample_bidir"
CUDA_VISIBLE_DEVICES="" $python scripts/plot_rgb_diff.py \
    --videos_path "$videos_path"
    # --no_source           # distance map only, no source frame on the left
    # --out_dir <dir>       # default: $videos_path/rgbdiff_viz