import numpy as np
from tqdm import tqdm
import cv2
from vbench.utils import load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

# Latent -> frame mapping assumed by the generator (1-indexed latent i):
#   i == 1        -> 1 frame
#   i > 1          -> frames [4i-6, 4i-3] (4 frames)
# 21 latents -> 81 frames.
FRAMES_PER_LATENT = 4
# Latents are produced in groups of 3; each group forms one chunk.
LATENT_GROUP_SIZE = 3

def get_frames(video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


def get_latent_frame_ranges(num_frames):
    """Split 0-indexed frame indices into per-latent ranges: latent 1 -> 1 frame,
    every following latent -> up to FRAMES_PER_LATENT frames."""
    ranges = [(0, 1)]
    frame = 1
    while frame < num_frames:
        end = min(frame + FRAMES_PER_LATENT, num_frames)
        ranges.append((frame, end))
        frame = end
    return ranges


def get_chunk_middle_pairs(num_frames):
    """Return the two middle 0-indexed frame indices for each chunk (a group of
    LATENT_GROUP_SIZE consecutive latents). Used for the in-chunk flickering score."""
    latent_ranges = get_latent_frame_ranges(num_frames)
    groups = [latent_ranges[i:i + LATENT_GROUP_SIZE] for i in range(0, len(latent_ranges), LATENT_GROUP_SIZE)]
    group_frame_ranges = [(g[0][0], g[-1][1]) for g in groups]

    pairs = []
    for start, end in group_frame_ranges:
        n = end - start
        if n < 2:
            continue
        mid = n // 2
        pairs.append((start + mid - 1, start + mid))
    return pairs


def get_chunk_boundary_pairs(num_frames):
    """Return (last_frame_of_prev_group, first_frame_of_next_group) index pairs
    for each boundary between consecutive groups of LATENT_GROUP_SIZE latents.
    Used for the interchunk flickering score."""
    latent_ranges = get_latent_frame_ranges(num_frames)
    groups = [latent_ranges[i:i + LATENT_GROUP_SIZE] for i in range(0, len(latent_ranges), LATENT_GROUP_SIZE)]
    group_frame_ranges = [(g[0][0], g[-1][1]) for g in groups]

    pairs = []
    for i in range(len(group_frame_ranges) - 1):
        prev_last_frame = group_frame_ranges[i][1] - 1
        next_first_frame = group_frame_ranges[i + 1][0]
        pairs.append((prev_last_frame, next_first_frame))
    return pairs


def calculate_mae(img1, img2):
    """Computing the mean absolute error (MAE) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


def mae_seq(frames, pairs_fn):
    ssds = []
    for prev_idx, next_idx in pairs_fn(len(frames)):
        ssds.append(calculate_mae(frames[prev_idx], frames[next_idx]))
    return np.array(ssds)


def cal_score(frames, pairs_fn):
    """please ensure the video is static"""
    score_seq = mae_seq(frames, pairs_fn)
    return (255.0 - np.mean(score_seq).item()) / 255.0


def cal_ratio_score(video_path):
    """Returns (1 - interchunk_score) / (1 - in_chunk_score) for a single video."""
    frames = get_frames(video_path)
    in_chunk_score = cal_score(frames, get_chunk_middle_pairs)
    interchunk_score = cal_score(frames, get_chunk_boundary_pairs)
    return (1.0 - interchunk_score) / (1.0 - in_chunk_score)


def temporal_flickering_ratio(video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        try:
            score_per_video = cal_ratio_score(video_path)
        except AssertionError:
            continue
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_temporal_flickering_ratio(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension='temporal_flickering_ratio', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = temporal_flickering_ratio(video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
