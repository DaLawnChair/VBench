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
        if key in VBENCH_16_DIMENSIONS:
            upload_data[VBENCH_16_DIMENSIONS_TO_TASK_INFO[key]] = value[0]
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
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load submission file')
    parser.add_argument('--file', type=str, required=True, help='Name of the evaluation file', default='evaluation_results')

    parser.add_argument('--file-replacement', type=str, help='Name of the evaluation file to replace a dimension with', default=None)
    parser.add_argument('--file-replacement-dims', nargs='*', type=str, help='Name of the evaluation file to replace a dimension with', default=None)


    # parser.add_argument('--model_name', type=str, required=True, help='Name of the model', default='t2v_model')
    args = parser.parse_args()

    upload_dict = submission(args.file)
    print(f"your submission info: \n{upload_dict} \n")

    if args.file_replacement is not None:
        upload_dict_replacement = submission(args.file_replacement)
    if args.file_replacement_dims is not None:
        for dim in args.file_replacement_dims:
            if VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict and VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict_replacement:
                upload_dict[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]] = upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]
                print(f"Replaced dimension {dim} with value {upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]} from file {args.file_replacement}")
            else:
                print(f"Warning: Dimension {dim} not found in one of the files, skipping replacement.")
    
    normalized_score = get_nomalized_score(upload_dict)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)
    print('+------------------|------------------+')
    print(f'|     quality score|{quality_score}|')
    print(f'|    semantic score|{semantic_score}|')
    print(f'|       total score|{final_score}|')
    print('+------------------|------------------+')


    # import ipdb;ipdb.set_trace()

    print(f"{final_score*100:.6f},{quality_score*100:.6f},{semantic_score*100:.6f},", end="")
    # print(f"{final_score}\t{quality_score}\t{semantic_score}\t")
    
    print(f"{ ','.join(f'{upload_dict[dim]*100:.6f}' for dim in TASK_INFO)}")

# python3 cal_final_no_zip_score.py \
#     --file-replacement-dims dynamic_degree \
#     --file /shared/john/AR_dawood/AR/vbench/rowwise_double_base_from_20k/iter_4000/sf_cf_extended//vbench_scores/full_sf_all_dims/results_2026-07-05-21:53:36_eval_results.json \
#     --file-replacement /shared/john/AR_dawood/AR/vbench/rowwise_double_base_from_20k/iter_4000/sf_cf_extended//vbench_scores/cf_dynamic_degree/results_2026-07-06-14:22:41_eval_results.json


# python3 cal_final_no_zip_score.py \
#     --file-replacement-dims dynamic_degree \
#     --file /shared/john/AR_dawood/AR/vbench/double_refine_gan/iter_7000/sf_cf_extended/seed0//vbench_scores/full_sf_all_dims/results_2026-07-06-04:57:03_eval_results.json \
#     --file-replacement /shared/john/AR_dawood/AR/vbench/double_refine_gan/iter_7000/sf_cf_extended/seed0//vbench_scores/cf_dynamic_degree/results_2026-07-06-14:22:44_eval_results.json


# python3 cal_final_no_zip_score.py \
#     --file-replacement-dims dynamic_degree \
#     --file /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended//vbench_scores/full_sf_all_dims/results_2026-07-05-21:52:49_eval_results.json \
#     --file-replacement /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended//vbench_scores/cf_dynamic_degree/results_2026-07-06-14:21:03_eval_results.json

# python3 cal_final_no_zip_score.py \
#     --file-replacement-dims dynamic_degree \
#     --file /shared/john/AR_dawood/Causal-Forcing/vbench/causal_forcing_dmd/sf_cf_extended//vbench_scores/full_sf_all_dims/results_2026-07-05-21:53:03_eval_results.json \
#     --file-replacement /shared/john/AR_dawood/Causal-Forcing/vbench/causal_forcing_dmd/sf_cf_extended//vbench_scores/cf_dynamic_degree/results_2026-07-06-14:28:58_eval_results.json