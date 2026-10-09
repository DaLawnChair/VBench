
device=7

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/_stage1_double_cache_14b_msm_continued_10k/videos_rsample_seed1"
CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
    --videos_path "$videos_path" \
    --mode custom_input_original \
    --output_path "$videos_path/cf_dynamic_degree_new" \
    --dimension dynamic_degree


videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k_bidirectional/videos_rsample_bidir"
CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
    --videos_path "$videos_path" \
    --mode custom_input_original \
    --output_path "$videos_path/cf_dynamic_degree_new" \
    --dimension dynamic_degree

videos_path="/mnt/ddn/george/diffusion/john-AR/vbench_results/261001_dawood-double_cache_14b_baseline_iter12k/videos_rsample"
CUDA_VISIBLE_DEVICES=$device python3 evaluate.py \
    --videos_path "$videos_path" \
    --mode custom_input_original \
    --output_path "$videos_path/cf_dynamic_degree_new" \
    --dimension dynamic_degree