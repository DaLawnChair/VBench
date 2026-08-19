import io
import os
import json
import shutil
import zipfile
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from constant import *

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


EXTRA_DIMENSIONS = { 'temporal_flickering_ratio', 'unitemp_flickering_ratio'}

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

    video_files = {}
    for key, value in data.items():
        if key in EXTRA_DIMENSIONS:
            upload_data[key] = value[0]
            video_files_dict[key] = value[1]
        elif key in VBENCH_16_DIMENSIONS:
            upload_data[key] = value[0]
            video_files_dict[key] = value[1]
        else:
            print(f"Warning: Key {key} not in VBENCH_16_DIMENSIONS, skipping.")

    # import ipdb; ipdb.set_trace()
    return upload_data, video_files

if __name__=="__main__":

    from pathlib import Path
    parser = argparse.ArgumentParser(description="Load evaluation files from a folder")
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing files ending with eval_results.json",
    )
    parser.add_argument(
            "--dest_folder",
            type=Path,
            required=True,
            help="Folder of end results",
        )
    parser.add_argument(
        "--num_of_samples",
        type=int,
        default=3
    )
    parser.add_argument(
        "--fixed_sample_list",
        nargs="+",
        type=int,
        default=[23, 45, 62]
    )
    parser.add_argument(
        "--get_best_worst",
        action="store_true"
    )
    parser.add_argument(
        "--is_split_video",
        action="store_true"
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
    video_files_dict = {}
    for eval_file in eval_files:
        upload_data, video_files = submission(eval_file)
        upload_dict.update(upload_data)
        video_files_dict.update(video_files)

    
    # def get_best_worse(dimension='scene', is_split_video=False):
    #     if is_split_video:
    #         full_clip_name = {video_files_dict[dimension][i]['video_name'].split('/')[-2] for i in range(len(video_files_dict[dimension]))}

    #         import ipdb;ipdb.set_trace()
    #         sorted_scores = sorted(video_files_dict[dimension], key=lambda item: item['video_results'])
    #     else:
    #         sorted_scores = sorted(video_files_dict[dimension], key=lambda item: item['video_results'])

    #     return list(sorted_scores.keys())[:args.num_of_samples], list(sorted_scores.keys())[-args.num_of_samples:]


    from collections import defaultdict
    from pathlib import Path
    import numpy as np


    def get_video_paths(dimension, sample_list=[23, 45, 62], is_split_video=False, get_best_worse=False):
        """
        Average scores belonging to the same video, then return the lowest-
        and highest-scoring video names.

        For split videos:
            /path/video_001/clip_0.mp4
            /path/video_001/clip_1.mp4
            -> grouped under "video_001"

        For non-split videos:
            /path/video_001.mp4
            -> grouped under "video_001.mp4"
        """
        grouped_scores = defaultdict(list)

        for item in video_files_dict[dimension]:
            # video_path = Path(item["video_path"])
            video_path = Path(item["video_path"]).expanduser().resolve()
            score = item["video_results"]

            if is_split_video:
                # Parent directory represents the original full video.
                common_video_name = "".join(str(video_path.parent).split('split_clip')) + '.mp4'
            else:
                # Group identical video filenames, ignoring directory.
                common_video_name = str(video_path)

            # Handles Python numbers, NumPy scalars, and similar scalar values.
            grouped_scores[common_video_name].append(float(score))

        averaged_scores = {
            video_name: float(np.mean(scores))
            for video_name, scores in grouped_scores.items()
        }
        all_samples = [{'video_path':video_name, 'video_results':averaged_scores[video_name]} for video_name in averaged_scores.keys()]
        set_sample_list = [all_samples[i] for i in sample_list]
                

        if get_best_worse:

            # Lowest score first, highest score last.
            sorted_scores = sorted(
                all_samples,
                key=lambda item: item['video_results'],
            )

            num_samples = min(args.num_of_samples, len(sorted_scores))

            worst = sorted_scores[:num_samples]
            best = sorted_scores[-num_samples:][::-1]

            # Return only the video names.
            # worst_names = [video_name for video_name, _ in worst]
            # best_names = [video_name for video_name, _ in best]

            return set_sample_list, worst, best
        return set_sample_list


    import cv2


    def write_score_on_video(
        input_path: str,
        output_path: str,
        score: float,
        label: str = "Score",
        font_scale: float = 1.0,
        thickness: int = 2,
        margin: int = 20,
    ) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        # mp4v is broadly supported, but does not preserve audio.
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

        text = f"{label}: {score:.4f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
        text_width, text_height = text_size

        # cv2.putText() uses the bottom-left corner of the text.
        x = max(margin, width - text_width - margin)
        y = margin + text_height

        frames_written = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Black outline for readability.
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 4,
                cv2.LINE_AA,
            )

            # White foreground text.
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frames_written += 1

        cap.release()
        writer.release()

        if frames_written == 0:
            raise RuntimeError(f"No frames were read from: {input_path}")

        print(
            f"Wrote {frames_written}/{num_frames} frames to {output_path}"
        )


# if __name__ == "__main__":
#     write_score_on_video(
#         input_path="input.mp4",
#         output_path="output_with_score.mp4",
#         score=0.9237,
#     )

    def save_video(video_path, dest_path):
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(video_path, dest_path)
        

    for dimension in upload_dict.keys():
        # sorted_scores = sorted(video_files_dict['scene'], key=lambda item: item['video_results'])
        # import ipdb;ipdb.set_trace()
        results = get_video_paths(dimension=dimension, is_split_video=args.is_split_video,
                                                                        get_best_worse=args.get_best_worst)
        if args.get_best_worst:
            set_sample_list, worst_videos, best_videos = results
        else:
            set_sample_list = results
            worst_videos = [] 
            best_videos = []

        for item in set_sample_list:
            
            end_dest_path = os.path.join(args.dest_folder, dimension, f"score_{item['video_results']:.4f}"  + os.path.basename(item['video_path']))

            # write_score_on_video(input_path=item['video_path'], output_path=end_dest_path, score=item['video_results'])
            save_video(video_path=item['video_path'], dest_path=end_dest_path)

        for item in worst_videos:
            # import ipdb;ipdb.set_trace()
            end_dest_path = os.path.join(args.dest_folder, dimension, 'worst', f"score_{item['video_results']:.4f}"  + os.path.basename(item['video_path']))
            # write_score_on_video(input_path=item['video_path'], output_path=end_dest_path, score=item['video_results'])
            save_video(video_path=item['video_path'], dest_path=end_dest_path)

        for item in best_videos:
            end_dest_path = os.path.join(args.dest_folder, dimension, 'best', f"score_{item['video_results']:.4f}"  + os.path.basename(item['video_path']))
            # write_score_on_video(input_path=item['video_path'], output_path=end_dest_path, score=item['video_results'])
            save_video(video_path=item['video_path'], dest_path=end_dest_path)


    # if args.file_replacement is not None:
    #     upload_dict_replacement = submission(args.file_replacement)
    # if args.file_replacement_dims is not None:
    #     for dim in args.file_replacement_dims:
    #         if VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict and VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim] in upload_dict_replacement:
    #             upload_dict[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]] = upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]
    #             print(f"Replaced dimension {dim} with value {upload_dict_replacement[VBENCH_16_DIMENSIONS_TO_TASK_INFO[dim]]} from file {args.file_replacement}")
    #         else:
    #             print(f"Warning: Dimension {dim} not found in one of the files, skipping replacement.")
    

# python3 grab_vbench_sampling.py --folder /shared/john/AR_dawood/infinity-rope/vbench_long/self_forcing_dmd_plus_inf_rope/sf_cf_extended/vbench_scores/full_sf_all_dims
# python3 grab_vbench_sampling.py \
# --folder /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims \
# --dest_folder /shared/john/AR_dawood/Causal-Forcing/vbench/self_forcing_dmd/sf_cf_extended/vbench_scores/full_sf_all_dims/sampled_videos/ \
# --get_best_worst 
## --is_split_video


