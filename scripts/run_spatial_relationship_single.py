#!/usr/bin/env python3
"""Run VBench's spatial_relationship metric on a single (prompt, video) pair.

Reimplements just the per-video body of spatial_relationship()/compute_spatial_relationship()
from vbench/spatial_relationship.py, without the json-driven batch/multi-rank machinery.
Must be run in the environment used for VBench's evaluate.py (vbench, detectron2, and the
GRIT checkpoint available).
"""
import argparse
import json
import os
import sys
import torch

VBENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VBENCH_ROOT not in sys.path:
    sys.path.insert(0, VBENCH_ROOT)

from torchvision import transforms
from vbench.utils import load_video, load_json
from vbench.third_party.grit_model import DenseCaptioning
from vbench.spatial_relationship import get_dect_from_grit, check_generate

CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR') or os.path.join(os.path.expanduser('~'), '.cache', 'vbench')
DEFAULT_MODEL_WEIGHT = f'{CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth'
DEFAULT_FULL_INFO = os.path.join(VBENCH_ROOT, 'vbench', 'VBench_full_info.json')
DEFAULT_FILENAME_MAPPING = os.path.join(VBENCH_ROOT, 'custom_small_vbench_set', 'sf_prompts_extended_to_base_special_prompts_mapping.json')


def parse_args():
    p = argparse.ArgumentParser(description="Run VBench's spatial_relationship dimension on one prompt/video")
    p.add_argument('--video', required=True, help='Path to the video file (.mp4/.gif/.png)')
    p.add_argument('--prompt', help="Base VBench prompt text (e.g. 'a bicycle on the left of a car, front view'), used to look up object_a/object_b/relationship in VBench_full_info.json. If omitted, it is derived from --video's filename via --filename-mapping-json.")
    p.add_argument('--device', default='cuda', help='Device for the GRIT model')
    p.add_argument('--model-weight', default=DEFAULT_MODEL_WEIGHT, help='Path to grit_b_densecap_objectdet.pth')
    p.add_argument('--full-info-json', default=DEFAULT_FULL_INFO, help='VBench_full_info.json to look up auxiliary_info from')
    p.add_argument('--filename-mapping-json', default=DEFAULT_FILENAME_MAPPING, help="Maps generated video filenames (truncated extended prompt + '-N.mp4') to base VBench prompts; used to derive --prompt from --video's filename when --prompt is not given")
    p.add_argument('--object-a', help='Override: skip both lookups and pass object_a directly')
    p.add_argument('--object-b', help='Override: skip both lookups and pass object_b directly')
    p.add_argument('--relationship', help='Override: skip both lookups and pass the relationship directly (e.g. "on the left of")')
    return p.parse_args()


def lookup_prompt_from_filename(filename_mapping_json, video_path):
    mapping = load_json(filename_mapping_json)
    basename = os.path.basename(video_path)
    if basename not in mapping:
        raise ValueError(f"Video filename not found in {filename_mapping_json}: {basename!r}")
    return mapping[basename]


def lookup_auxiliary_info(full_info_json, prompt):
    for entry in load_json(full_info_json):
        if entry.get('prompt_en') == prompt and 'spatial_relationship' in entry.get('dimension', []):
            return entry['auxiliary_info']['spatial_relationship']['spatial_relationship']
    raise ValueError(f"Prompt not found under dimension 'spatial_relationship' in {full_info_json}: {prompt!r}")


def main():
    args = parse_args()

    if args.object_a and args.object_b and args.relationship:
        object_info = {'object_a': args.object_a, 'object_b': args.object_b, 'relationship': args.relationship}
    else:
        prompt = args.prompt or lookup_prompt_from_filename(args.filename_mapping_json, args.video)
        object_info = lookup_auxiliary_info(args.full_info_json, prompt)

    model = DenseCaptioning(device = torch.device(args.device))
    model.initialize_model_det(model_weight=args.model_weight)

    # Same preprocessing as spatial_relationship() in vbench/spatial_relationship.py
    video_tensor = load_video(args.video, num_frames=16)
    _, _, h, w = video_tensor.size()
    if min(h, w) > 768:
        scale = 720. / min(h, w)
        video_tensor = transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)

    predictions = get_dect_from_grit(model, video_tensor.permute(0, 2, 3, 1))
    frame_scores = check_generate(object_info, predictions)
    success_rate = sum(frame_scores) / len(frame_scores)

    # Same shape as the per-video dicts compute_spatial_relationship() returns in video_results
    result = {
        'prompt': prompt,
        'video_path': args.video,
        'video_results': success_rate,
        'frame_results': frame_scores,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
