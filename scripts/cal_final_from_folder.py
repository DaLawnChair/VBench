import io
import os
import json
import zipfile
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constant import *

VBENCH_16_DIMENSIONS = [
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "human_action",
        "color",
        "spatial_relationship",
        "scene",
        "temporal_style",
        "appearance_style",
        "overall_consistency",
    ]

VBENCH_16_DIMENSIONS_TO_TASK_INFO = {
    "subject_consistency": "subject consistency",
    "background_consistency": "background consistency",
    "temporal_flickering": "temporal flickering",
    "motion_smoothness": "motion smoothness",
    "dynamic_degree": "dynamic degree",
    "aesthetic_quality": "aesthetic quality",
    "imaging_quality": "imaging quality",
    "object_class": "object class",
    "multiple_objects": "multiple objects",
    "human_action": "human action",
    "color": "color",
    "spatial_relationship": "spatial relationship",
    "scene": "scene",
    "temporal_style": "temporal style",
    "appearance_style": "appearance style",
    "overall_consistency": "overall consistency",
}

# EXTRA_DIMENSIONS = { 'temporal_flickering_ratio', 'unitemp_flickering_ratio'}
EXTRA_DIMENSIONS = set()


def submission(file):
    # os.makedirs(model_name, exist_ok=True)
    # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    #     zip_ref.extractall(model_name)
    upload_data = {}
    # load your score
    # for file in os.listdir(model_name):
    #     if file.startswith('.') or file.startswith('__'):
    #         print(f"Skip the file: {file}")
    #         continue
    #     cur_file = os.path.join(model_name, file)
    #     if os.path.isdir(cur_file):
    #         for subfile in os.listdir(cur_file):
    #             if subfile.endswith(".json"):
    #                 with open(os.path.join(cur_file, subfile)) as ff:
    #                     cur_json = json.load(ff)
    #                     if isinstance(cur_json, dict):
    #                         for key in cur_json:
    #                             upload_data[key.replace('_',' ')] = cur_json[key][0]
    #     elif cur_file.endswith('json'):
    #         with open(cur_file) as ff:
    #             cur_json = json.load(ff)
    #             if isinstance(cur_json, dict):
    #                 for key in cur_json:
    #                     upload_data[key.replace('_',' ')] = cur_json[key][0]
        
    #     for key in TASK_INFO:
    #         if key not in upload_data:
    #             upload_data[key] = 0

    with open(file, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        aggregate = value[0]
        if isinstance(aggregate, list) and len(value) >= 2 and isinstance(value[1], list):
            # [metric_names, metric_values, per_video]: dynamic_degree with the
            # diff-of-diff metrics; the dimension's own score is the first value.
            aggregate = value[1][aggregate.index(key)] if key in aggregate else value[1][0]
            extra = {n: v for n, v in zip(value[0], value[1]) if n != key}
            if extra:
                print(f"{key} extra metrics (not part of the VBench score): {extra}")
        if key in EXTRA_DIMENSIONS:
            upload_data[key] = aggregate
        if key in VBENCH_16_DIMENSIONS:
            upload_data[VBENCH_16_DIMENSIONS_TO_TASK_INFO[key]] = aggregate
        else:
            print(f"Warning: Key {key} not in VBENCH_16_DIMENSIONS, skipping.")

    # import ipdb; ipdb.set_trace()
    return upload_data

def get_nomalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in TASK_INFO:
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        if key not in upload_data:
            print(f"Warning: Key {key} not found in upload_data. Setting normalized score to 0.")
            normalized_score[key] = 0
            continue
        else:
            normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
            normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
            print(f"Normalized score for {key}: {normalized_score[key]} (raw: {upload_data[key]}, min: {min_val}, max: {max_val}, weight: {DIM_WEIGHT[key]})")
    return normalized_score

def get_quality_score(normalized_score):
    quality_score = []
    for key in QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score)/sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    return quality_score

def get_semantic_score(normalized_score):
    semantic_score = []
    for key in SEMANTIC_LIST:
        semantic_score.append(normalized_score[key])
    semantic_score  = sum(semantic_score)/sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST ])
    return semantic_score

def get_final_score(quality_score,semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)


# if __name__ == "__main__":
#     from pathlib import Path
#     parser = argparse.ArgumentParser(description="Load evaluation files from a folder")
#     parser.add_argument(
#         "--folder",
#         type=Path,
#         required=True,
#         help="Folder containing files ending with eval_results.json",
#     )
#     args = parser.parse_args()

#     import ipdb;ipdb.set_trace()
#     folder = args.folder

#     if not folder.is_dir():
#         parser.error(f"Folder does not exist or is not a directory: {folder}")

#     eval_files = sorted(
#         folder.glob("*eval_results.json"),
#         key=lambda path: path.name,
#     )

#     if not eval_files:
#         parser.error(
#             f'No files ending with "eval_results.json" found in {folder}'
#         )

#     for eval_file in eval_files:
#         print(f"Processing: {eval_file}")

#         upload_dict = submission(eval_file)

#         normalized_score = get_nomalized_score(upload_dict)
#         quality_score = get_quality_score(normalized_score)
#         semantic_score = get_semantic_score(normalized_score)
#         final_score = get_final_score(quality_score, semantic_score)

#         print(f"Quality score:  {quality_score}")
#         print(f"Semantic score: {semantic_score}")
#         print(f"Final score:    {final_score}")

if __name__=="__main__":

    from pathlib import Path
    parser = argparse.ArgumentParser(description="Load evaluation files from a folder")
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing files ending with eval_results.json",
    )
    args = parser.parse_args()


    from pathlib import Path
    folder = Path(args.folder)
    eval_files = sorted(
        folder.glob("*eval_results.json"),
        key=lambda path: path.name,
    )
    print(eval_files)
    upload_dict = {}
    for eval_file in eval_files:
        upload_dict.update(submission(eval_file))
    
    # upload_dict = submission(folder)
    print(f"your submission info: \n{upload_dict} \n")

    # Fail before opening final_scores.txt: a missing dimension would otherwise
    # be scored as 0 in the quality/semantic totals and then crash mid-write.
    missing = [dim for dim in TASK_INFO if dim not in upload_dict]
    if missing:
        sys.exit(f"ERROR: {folder} is missing eval_results for dimension(s): "
                 f"{', '.join(missing)}. Re-run evaluate.py for them; "
                 f"final_scores.txt not written.")

    # if args.file_replacement is not None:
    #     upload_dict_replacement = submission(args.file_replacement)
    # if args.file_replacement_dims is not None:
    #     for dim in args.file_replacement_dims:
    #         if VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict and VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict_replacement:
    #             upload_dict[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]] = upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]
    #             print(f"Replaced dimension {dim} with value {upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]} from file {args.file_replacement}")
    #         else:
    #             print(f"Warning: Dimension {dim} not found in one of the files, skipping replacement.")
    
    normalized_score = get_nomalized_score(upload_dict)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)
    out_path = folder / "final_scores.txt"
    out_f = open(out_path, "w")
    def tee(*a, **kw):
        print(*a, **kw)
        print(*a, **kw, file=out_f)
    tee('+------------------|------------------+')
    tee(f'|     quality score|{quality_score}|')
    tee(f'|    semantic score|{semantic_score}|')
    tee(f'|       total score|{final_score}|')
    tee('+------------------|------------------+')


#     # import ipdb;ipdb.set_trace()

    tee(f"{final_score*100:.6f},{quality_score*100:.6f},{semantic_score*100:.6f},", end="")
    # print(f"{final_score}\t{quality_score}\t{semantic_score}\t")
    
    tee(f"{ ','.join(f'{upload_dict[dim]*100:.6f}' for dim in TASK_INFO)}")
    tee(f"Normalized_score: { ','.join(f'{normalized_score[dim]*100:.6f}' for dim in TASK_INFO)}")
 
    tee("extra dimensions:")
    tee(f"{ ','.join(f'{upload_dict[dim]:.6f}' for dim in EXTRA_DIMENSIONS)}")
    out_f.close()
    print(f"Saved to {out_path}")
    

    # python3 scripts/cal_final_from_folder.py --folder /mnt/ddn/george/diffusion/john-AR/vbench_results/260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter500/double_cache/vbench_scores/full_sf_all_dims
    # python3 scripts/cal_final_from_folder.py --folder /mnt/ddn/george/diffusion/john-AR/vbench_results/260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter750/double_cache/vbench_scores/full_sf_all_dims
    # python3 scripts/cal_final_from_folder.py --folder /mnt/ddn/george/diffusion/john-AR/vbench_results/260828_dc_gan_dmd_msm_ckpt_gan_low_renoise_150_250_dmd_standard_and_temporal_regularization_on_step_1_only_gpu2/iter1000/double_cache/vbench_scores/full_sf_all_dims
