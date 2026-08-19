import numpy as np
from tqdm import tqdm
import cv2
from vbench.utils import load_dimension_info

import torch
from typing import Dict, Optional, Union

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


# def get_latent_frame_ranges(num_frames):
#     """Split 0-indexed frame indices into per-latent ranges: latent 1 -> 1 frame,
#     every following latent -> up to FRAMES_PER_LATENT frames."""
#     ranges = [(0, 1)]
#     frame = 1
#     while frame < num_frames:
#         end = min(frame + FRAMES_PER_LATENT, num_frames)
#         ranges.append((frame, end))
#         frame = end
#     return ranges


# def get_chunk_middle_pairs(num_frames):
#     """Return the two middle 0-indexed frame indices for each chunk (a group of
#     LATENT_GROUP_SIZE consecutive latents). Used for the in-chunk flickering score."""
#     latent_ranges = get_latent_frame_ranges(num_frames)
#     groups = [latent_ranges[i:i + LATENT_GROUP_SIZE] for i in range(0, len(latent_ranges), LATENT_GROUP_SIZE)]
#     group_frame_ranges = [(g[0][0], g[-1][1]) for g in groups]

#     pairs = []
#     for start, end in group_frame_ranges:
#         n = end - start
#         if n < 2:
#             continue
#         mid = n // 2
#         pairs.append((start + mid - 1, start + mid))
#     return pairs


# def get_chunk_boundary_pairs(num_frames):
#     """Return (last_frame_of_prev_group, first_frame_of_next_group) index pairs
#     for each boundary between consecutive groups of LATENT_GROUP_SIZE latents.
#     Used for the interchunk flickering score."""
#     latent_ranges = get_latent_frame_ranges(num_frames)
#     groups = [latent_ranges[i:i + LATENT_GROUP_SIZE] for i in range(0, len(latent_ranges), LATENT_GROUP_SIZE)]
#     group_frame_ranges = [(g[0][0], g[-1][1]) for g in groups]

#     pairs = []
#     for i in range(len(group_frame_ranges) - 1):
#         prev_last_frame = group_frame_ranges[i][1] - 1
#         next_first_frame = group_frame_ranges[i + 1][0]
#         pairs.append((prev_last_frame, next_first_frame))
#     return pairs


# def calculate_mae(img1, img2):
#     """Computing the mean absolute error (MAE) between two images."""
#     if img1.shape != img2.shape:
#         print("Images don't have the same shape.")
#         return
#     return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


# def mae_seq(frames, pairs_fn):
#     ssds = []
#     for prev_idx, next_idx in pairs_fn(len(frames)):
#         ssds.append(calculate_mae(frames[prev_idx], frames[next_idx]))
#     return np.array(ssds)


# def cal_score(frames, pairs_fn):
#     """please ensure the video is static"""
#     score_seq = mae_seq(frames, pairs_fn)
#     return (255.0 - np.mean(score_seq).item()) / 255.0



def flickering_ratio(
    video: np.ndarray,
    frames_per_latent: int = 4,
    first_frame_offset: bool = False,
    eps: float = 1e-8,
    channel_last: bool = True,
    block_latents: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Flickering Ratio from decoded video frames.

    For Wan-style VAEs:
        latent_T = 1 + (frame_T - 1) // 4

    Mapping:
        z0 -> x0
        z1 -> x1, x2, x3, x4
        z2 -> x5, x6, x7, x8
        ...

    Therefore, normal latent boundaries are:
        x4 -> x5
        x8 -> x9
        x12 -> x13
        ...

    Args:
        video:
            [B, C, T, H, W] if channel_first=True
            [B, T, C, H, W] if channel_first=False

        frames_per_latent:
            Temporal VAE compression factor. Usually 4.

        first_frame_offset:
            Use True for Wan-style VAE decoding where z0 corresponds to x0
            and later latents decode groups of 4 frames.

        block_latents:
            Optional number of latent frames per generated block.
            If provided, computes intra-block and inter-block FR.

    Returns:
        Dictionary with overall FR, per-boundary FR, and boundary indices.
    """
    video = np.asarray(video)
    if video.ndim == 4:
        video = np.expand_dims(video, axis=0)

    if channel_last:
        # [B, C, T, H, W] -> [B, T, C, H, W]
        # video = video.permute(0, 2, 1, 3, 4)
        
        # [B, T, H, W, C] -> [B, T, C, H, W]
        video = np.transpose(video, (0, 1, 4, 2, 3))


    B, T, C, H, W = video.shape

    if first_frame_offset:
        # Wan-style:
        # boundaries are x4->x5, x8->x9, x12->x13, ...
        boundary_idx = np.arange(
            frames_per_latent,
            T - 1,
            frames_per_latent,
            dtype=np.int64,
        )
    else:
        # Non-offset formulation:
        # boundaries are x3->x4, x7->x8, x11->x12, ...
        boundary_idx = np.arange(
            frames_per_latent - 1,
            T - 1,
            frames_per_latent,
            dtype=np.int64,
        )

    if len(boundary_idx) == 0:
        raise ValueError(f"No valid latent boundaries found for video length T={T}")

    prev_local = video[:, boundary_idx - 1]  # x_{boundary-1}
    boundary = video[:, boundary_idx]       # x_boundary
    next_cross = video[:, boundary_idx + 1] # x_{boundary+1}

    reduce_dims = (2, 3, 4)  # C, H, W

    numerator = np.abs(next_cross - boundary).sum(axis=reduce_dims)
    denominator = np.abs(boundary - prev_local).sum(axis=reduce_dims)

    fr = numerator / np.maximum(denominator,eps)

    out: Dict[str, Union[np.ndarray, np.floating]] = {
        "fr_mean": np.mean(fr),
        "fr_per_boundary": fr,
        "boundary_frame_indices": boundary_idx,
    }

    if block_latents is not None:
        if first_frame_offset:
            # boundary_idx=4 means boundary between z1 and z2
            latent_idx = boundary_idx // frames_per_latent
        else:
            # boundary_idx=3 means boundary between z0 and z1
            latent_idx = boundary_idx // frames_per_latent

        same_block = (latent_idx // block_latents) == ((latent_idx + 1) // block_latents)

        out["intra_block_mean"] = (
            np.mean(fr[:, same_block])
            if np.any(same_block)
            else np.float64(np.nan)
        )

        out["inter_block_mean"] = (
            np.mean(fr[:, ~same_block])
            if np.any(~same_block)
            else np.float64(np.nan)
        )

        out["intra_block_mask"] = same_block
        out["inter_block_mask"] = ~same_block

    # import ipdb;ipdb.set_trace()
    return out

def cal_ratio_score(video_path):
    """Returns (1 - interchunk_score) / (1 - in_chunk_score) for a single video."""
    frames = get_frames(video_path)
    # in_chunk_score = cal_score(frames, get_chunk_middle_pairs)
    # interchunk_score = cal_score(frames, get_chunk_boundary_pairs)
    output = flickering_ratio(frames,block_latents=3,first_frame_offset=True)
    return output['inter_block_mean']/output['intra_block_mean']
    # return (1.0 - output['inter_block_mean']) / (1.0 - output['intra_block_mean'])

    # return (1.0 - interchunk_score) / (1.0 - in_chunk_score)


def unitemp_flickering_ratio(video_list):
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


def compute_unitemp_flickering_ratio(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension='unitemp_flickering_ratio', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = unitemp_flickering_ratio(video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
