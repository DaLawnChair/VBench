# Notable changes:
Within VBench, the default custom prompts setting runs the same set of dimensions on all videos based off of the video's file name as its prompt and is not supported for dimenisons with auxillary information like "color". The modifications here are mainly for accomidation to custom videos with mapping of filename to full prompt to better match the "vbench standard" inference pattern.

Notably the custom datasets are mostly modifications to the standard vbench propts, but LLM extended. This is the version standardized by Self-Forcing. When generating these videos, the prompts are too long to fit within the standard filename size, so a mapping of shortended filenames to the full prompt is used. The custom prompts are then mapped to its corresponding dimension and saved.

# 5s generation eval
For 5s generation, use `run_sf_cf_eval_sleep.sh`, which will wait for the inference of a video directory to finish (aka get to 1046 files), and then run the vbench eval on that set across all dimesions in the Self-Forcing prompts (~946 videos) and then after the Causal-Forcing prompts (100 videos, only on the dynamic degree section).

# 30s+ generation eval
For long generation, use `run_sf_cf_eval_long_6.sh`, which will evaluate over all dimensions for the Self-Forcing Vbench promptset (946). Note that the dimension 'background_consistency' requires extra packages that the `vbench` environment does not have. Also note that the start of the eval requires splitting the videos into segments, which can take a few hours and doens't use the GPU.
