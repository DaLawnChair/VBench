
# conda activate vbench
export VBENCH_CACHE_DIR="/mnt/ddn/george/VBench/pretrained"

device="5"

# video_parent_dir="/mnt/ddn/george/diffusion/john-AR/vbench_results/"
# exp_name="260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter500/double_cache"
# video="A front view of a car parked on the right side of the frame, with a bicycle positioned to the left of the car. The car is a sleek modern sedan with po-0.mp4"
# CUDA_VISIBLE_DEVICES="${device}" python3 scripts/run_spatial_relationship_single.py \
#   --video "${video_parent_dir}/${exp_name}/${video}" \
#   --device "cuda:${device}"

# video_parent_dir="/mnt/ddn/george/diffusion/john-AR/vbench_results/"
# exp_name="260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter500/double_cache"
# video="A front view of a car positioned to the right of a motorcycle on a clean, empty road. The car is a sleek, modern sedan with polished metallic paint, a-0.mp4"
# CUDA_VISIBLE_DEVICES="${device}" python3 scripts/run_spatial_relationship_single.py \
#   --video "${video_parent_dir}/${exp_name}/${video}" \
#   --device "cuda:${device}"

# video_parent_dir="/mnt/ddn/george/diffusion/john-AR/vbench_results/"
# exp_name="260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter500/double_cache"
# video="A front view of a bus positioned to the right of a traffic light. The bus is painted in a standard city transit color scheme with advertisements on th-0.mp4"
# CUDA_VISIBLE_DEVICES="${device}" python3 scripts/run_spatial_relationship_single.py \
#   --video "${video_parent_dir}/${exp_name}/${video}" \
#   --device "cuda:${device}"

# video_parent_dir="/mnt/ddn/george/diffusion/john-AR/vbench_results/"
# exp_name="260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter500/double_cache"
# video="A front view of a fire hydrant positioned on the right side of the frame, with a traffic light prominently displayed to its left. The traffic light sh-0.mp4"
# CUDA_VISIBLE_DEVICES="${device}" python3 scripts/run_spatial_relationship_single.py \
#   --video "${video_parent_dir}/${exp_name}/${video}" \
#   --device "cuda:${device}"

video_parent_dir="/mnt/ddn/george/diffusion/john-AR/vbench_results/"
exp_name="_references/sample_videos/260828_mirror"
video="A front view of a car parked on the right side of the frame, with a bicycle positioned to the left of the car. The car is a sleek modern sedan with po-0.mp4"
CUDA_VISIBLE_DEVICES="${device}" python3 scripts/run_spatial_relationship_single.py \
  --video "${video_parent_dir}/${exp_name}/${video}" \
  --device "cuda:${device}"

# anyway, please run a ckpt 0828 / 0901 on a few prompts like this

# 1. a car on the right of a motorcycle, front view → object_a=car, object_b=motorcycle, relationship=on the right of
#    A front view of a car positioned to the right of a motorcycle on a clean, empty road. The car is a sleek, modern sedan with polished metallic paint, a-0.mp4
# 2. a motorcycle on the left of a bus, front view → object_a=motorcycle, object_b=bus, relationship=on the left of
#    A front view of a motorcycle positioned to the left of a large bus. The motorcycle has a sleek design with polished metallic surfaces and a shiny helm-0.mp4
# 3. a bus on the right of a traffic light, front view → object_a=bus, object_b=traffic light, relationship=on the right of
#    A front view of a bus positioned to the right of a traffic light. The bus is painted in a standard city transit color scheme with advertisements on th-0.mp4
# 4. a traffic light on the left of a fire hydrant, front view → object_a=traffic light, object_b=fire hydrant, relationship=on the left of
#    A front view of a fire hydrant positioned on the right side of the frame, with a traffic light prominently displayed to its left. The traffic light sh-0.mp4
# 5. a fire hydrant on the right of a stop sign, front view → object_a=fire hydrant, object_b=stop sign, relationship=on the right of
#    A front view of a fire hydrant positioned to the right of a stop sign. The fire hydrant is bright red with a reflective collar and nozzle handle. The -0.mp4

# Examples: 
# A close-up front view of a cat positioned to the right of a dog. Both animals are facing forward, with the cat looking slightly towards the camera, ex-0.mp4
# A close-up front view of a sleek digital alarm clock placed on a clean, flat surface. To the right of the clock lies a modern, rectangular TV remote c-0.mp4
# A close-up front view of a wine glass placed on the left side, with a bottle positioned slightly behind and to the right of the glass. The wine glass -0.mp4
# A dynamic front view of a modern train positioned to the right of a traditional wooden boat, emphasizing the contrast between old and new. The train i-0.mp4
# A front view of a bottle placed to the right of a bowl. The bowl is positioned slightly behind the bottle, creating a clear spatial relationship betwe-0.mp4
# A front view of a bus positioned to the right of a traffic light. The bus is painted in a standard city transit color scheme with advertisements on th-0.mp4
# A front view of a car positioned to the right of a motorcycle on a clean, empty road. The car is a sleek, modern sedan with polished metallic paint, a-0.mp4
# A front view of a fire hydrant positioned to the right of a stop sign. The fire hydrant is bright red with a reflective collar and nozzle handle. The -0.mp4
# A front view of a wine glass positioned to the right of a cup. Both items are centered in the frame, with the wine glass slightly overlapping the cup.-0.mp4
# A still front view of a vase placed to the right of a pair of scissors. The vase is tall and elegant, made of clear glass with intricate etchings, ref-0.mp4