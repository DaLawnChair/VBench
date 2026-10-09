#!/usr/bin/env bash
# Visualise RAFT optical-flow fields (as used by dynamic_degree) for a folder of videos.
# Writes <videos_path>/flow_viz/<video>_flow.mp4 + .csv  (see scripts/plot_flow_fields.py --help).
# Usage: bash run_plot_flow_fields.sh [videos_path]   (defaults to the path below)
set -euo pipefail
cd "$(dirname "$0")"

device=6
python=/shared/miniconda3/envs/vbench/bin/python

export VBENCH_CACHE_DIR=/mnt/ddn/george/VBench/pretrained   # RAFT ckpt: $VBENCH_CACHE_DIR/raft_model/models/raft-things.pth

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k/videos_rsample"
CUDA_VISIBLE_DEVICES=$device $python scripts/plot_flow_fields.py \
    --videos_path "$videos_path" \
    --side_by_side
    # --flow_scale 10        # fixed colour scale (px) across videos instead of per-video max
    # --vbench_subsample     # reproduce the original ~8 fps get_frames() resampling
    # --out_dir <dir>        # default: $videos_path/flow_viz

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k_bidirectional/videos_rsample_bidir"
CUDA_VISIBLE_DEVICES=$device $python scripts/plot_flow_fields.py \
    --videos_path "$videos_path" \
    --side_by_side
    # --flow_scale 10        # fixed colour scale (px) across videos instead of per-video max
    # --vbench_subsample     # reproduce the original ~8 fps get_frames() resampling
    # --out_dir <dir>        # default: $videos_path/flow_viz