# folders=(
# /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/Causal-Forcing/vbench/causal_forcing_dmd/sf_cf_extended//vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/seed0/
# /shared/john/AR_dawood/Causal-Forcing/vbench/causal_forcing_dmd/sf_cf_extended//vbench_scores/full_sf_all_dims/seed0/
# /shared/john/AR_dawood/AR/vbench/rowwise_double_base_from_20k/iter_4000/sf_cf_extended//vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/AR/vbench/double_refine_gan/iter_7000/sf_cf_extended/seed0//vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_10000/sf_cf_extended/vbench_scores/full_sf_all_dims
# /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_37000/sf_cf_extended/vbench_scores/full_sf_all_dims
# /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_37000/sf_cf_extended/vbench_scores/full_sf_all_dims/seed0/
# /shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_24000/sf_cf_extended/vbench_scores/full_sf_all_dims
# /shared/john/AR_dawood/AR/vbench/double_cache_bidir_first_critic_match/iter_25000/sf_cf_extended/vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/AR/vbench/double_cache_refine_dual_dmd_base_14b/iter_9000/sf_cf_extended/vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/AR/vbench/double_gan_dmd_1_step_14b/iter_1000/sf_cf_extended/vbench_scores/full_sf_all_dims/
# /shared/john/AR_dawood/AR/vbench/double_gan_dmd_1_step_14b_37k/iter_1500/sf_cf_extended/vbench_scores/full_sf_all_dims/
# )


folders=(
/shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/seed0/
/shared/john/AR_dawood/Causal-Forcing/vbench/causal_forcing_dmd/sf_cf_extended//vbench_scores/full_sf_all_dims/seed0/
/shared/john/AR_dawood/AR/vbench/double_cache_14b_baseline/iter_37000/sf_cf_extended/vbench_scores/full_sf_all_dims/seed0/
)

for folder in ${folders[@]}
do
    result="${folder#*/vbench/}"      # Remove everything through /vbench/
    result="${result%%/sf_cf_extended/*}"  # Remove everything after iter_xxxx

    python3 grab_vbench_sampling.py \
    --folder "$folder" \
    --dest_folder view_videos_seed0/${result} \
    --get_best_worst 
    # --is_split_video


    # echo "$folder"

    # echo cp -r $folder view_videos/${result}
done