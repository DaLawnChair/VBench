#!/usr/bin/env python3
"""Visualise the RAFT optical-flow fields behind VBench's dynamic_degree.

For every input video with n frames this writes an (n-1)-frame video in which
frame i is the colour-coded flow field `flow_up` between source frames
(f0, f1) = (i, i+1), exactly as computed inside vbench/dynamic_degree.py
(same RAFT checkpoint, same padding, iters=20, test_mode=True).  Each frame
carries a 50%-opacity box with the pair's `max_rad` (mean of the top-5% flow
magnitudes, the per-pair score dynamic_degree thresholds) and whether the pair
straddles a chunk border, a latent-frame border, or neither.

Frame layout assumed (Wan2.1 VAE, 4x temporal compression):
  latent 0 -> pixel frame 0, latent k -> pixel frames 4k-3 .. 4k
  latent border k|k+1 lies between pixel frames 4k and 4k+1
  chunk border = latent border with (k+1) % num_frame_per_block == 0

By default every pixel frame is used (pair i = frames i,i+1).  With
--vbench_subsample the original get_frames() behaviour (resample to ~8 fps)
is reproduced instead and the border labels account for the stride.

Colour coding: hue = flow direction, saturation = magnitude (Baker/Scharstein
colour wheel, same as RAFT's flow_viz).  Magnitude is normalised by a fixed
per-video scale so frames are comparable; default scale = the largest max_rad
in the video.  A colour-wheel legend is drawn bottom-right.

Run from anywhere; only needs torch, opencv, easydict (no decord / full vbench deps).

Example:
  VBENCH_CACHE_DIR=/mnt/ddn/george/VBench/pretrained \\
  CUDA_VISIBLE_DEVICES=6 python3 scripts/plot_flow_fields.py \\
      --videos_path /mnt/ddn/george/diffusion/john-AR/vbench_results/_stage1_double_cache_14b_msm_continued_10k/videos_rsample_seed1
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# import RAFT straight from third_party so that vbench/__init__.py (decord, clip, ...) is not pulled in
sys.path.insert(0, os.path.join(ROOT, "vbench", "third_party"))

from RAFT.core.raft import RAFT  # noqa: E402
from RAFT.core.utils_core.utils import InputPadder  # noqa: E402
from RAFT.core.utils_core.flow_viz import flow_uv_to_colors  # noqa: E402

# same resolution as vbench.utils.CACHE_DIR
CACHE_DIR = os.environ.get("VBENCH_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "vbench")


# --------------------------------------------------------------------------- #
# pieces copied from vbench/dynamic_degree.py so the numbers match exactly
# --------------------------------------------------------------------------- #
def load_model(model_path, device):
    args = edict({"model": model_path, "small": False, "mixed_precision": False, "alternate_corr": False})
    model = RAFT(args)
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    model.to(device)
    model.eval()
    return model


def get_score(flo):
    """flo: [1,2,H,W] torch -> (max_rad, flow [H,W,2] numpy). Same statistic as DynamicDegree.get_score."""
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    rad = np.sqrt(np.square(flo[:, :, 0]) + np.square(flo[:, :, 1]))
    h, w = rad.shape
    cut_index = int(h * w * 0.05)
    max_rad = np.mean(np.abs(np.sort(-rad.flatten()))[:cut_index])
    return float(max_rad), flo


def vbench_threshold(frame, count):
    """DynamicDegree.set_params: motion threshold for one pair, and required #pairs."""
    scale = min(list(frame.shape)[-2:])
    return 6.0 * (scale / 256.0), round(4 * (count / 16.0))


def read_frames(video_path, device, interval):
    """Return (list of [1,3,H,W] float tensors on device, list of kept pixel-frame indices, fps)."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames, bgr = [], []
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break
        bgr.append(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()[None].to(device))
    video.release()
    assert frames, f"no frames decoded from {video_path}"
    kept = list(range(0, len(frames), interval))
    return [frames[i] for i in kept], [bgr[i] for i in kept], kept, fps


# --------------------------------------------------------------------------- #
# border bookkeeping
# --------------------------------------------------------------------------- #
def classify_pair(f0, f1, num_frame_per_block):
    """Which latent borders (between pixel 4k and 4k+1) lie inside [f0, f1)?"""
    ks = [k for k in range(f0 // 4, f1 // 4 + 1) if f0 <= 4 * k < f1]
    chunk = [k for k in ks if (k + 1) % num_frame_per_block == 0]
    if chunk:
        k = chunk[0]
        return "chunk", f"CHUNK BORDER  latent {k}|{k + 1}  (chunk {(k + 1) // num_frame_per_block - 1}|{(k + 1) // num_frame_per_block})"
    if ks:
        k = ks[0]
        return "latent", f"latent border  latent {k}|{k + 1}"
    return "none", "intra-latent (no border)"


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
LABEL_COLOURS = {"chunk": (80, 80, 255), "latent": (60, 230, 255), "none": (230, 230, 230)}  # BGR
LABEL_THICKNESS = {"chunk": 2, "latent": 1, "none": 1}


def flow_to_bgr(flow, scale):
    u, v = flow[..., 0] / scale, flow[..., 1] / scale
    rad = np.sqrt(u * u + v * v)
    over = rad > 1.0
    u = np.where(over, u / np.maximum(rad, 1e-6), u)
    v = np.where(over, v / np.maximum(rad, 1e-6), v)
    return flow_uv_to_colors(u, v, convert_to_bgr=True)


def colour_wheel(size):
    ys, xs = np.mgrid[0:size, 0:size]
    u = (xs - size / 2) / (size / 2)
    v = (ys - size / 2) / (size / 2)
    rad = np.sqrt(u * u + v * v)
    img = flow_uv_to_colors(np.clip(u, -1, 1), np.clip(v, -1, 1), convert_to_bgr=True)
    img[rad > 1] = 0
    return img, rad <= 1


def draw_overlay(img, lines, label_kind, wheel=None, scale=None):
    img = img.copy()
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    pad, lh = 8, 22
    widths = [cv2.getTextSize(t, font, fs, th)[0][0] for t, _ in lines]
    box_w, box_h = max(widths) + 2 * pad, lh * len(lines) + 2 * pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)  # 50% opacity box
    for n, (text, colour) in enumerate(lines):
        thick = LABEL_THICKNESS[label_kind] if n == len(lines) - 1 else th
        cv2.putText(img, text, (pad, pad + lh * (n + 1) - 6), font, fs, colour, thick, cv2.LINE_AA)
    if wheel is not None:
        wimg, mask = wheel
        s = wimg.shape[0]
        h, w = img.shape[:2]
        y0, x0 = h - s - 8, w - s - 8
        roi = img[y0:y0 + s, x0:x0 + s]
        roi[mask] = (0.7 * wimg[mask] + 0.3 * roi[mask]).astype(np.uint8)
        cv2.putText(img, f"{scale:.1f}px", (x0 - 2, y0 - 6), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to_h264(src, dst):
    """Transcode to H.264 / yuv420p (+faststart) so the file plays in VS Code, browsers, etc."""
    if shutil.which("ffmpeg") is None:
        print("   WARNING: ffmpeg not found, leaving MPEG-4 part 2 output (may not play in VS Code)")
        os.replace(src, dst)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p needs even dims
           "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)
    os.remove(src)


# --------------------------------------------------------------------------- #
def process_video(video_path, model, device, args, wheel):
    interval = 1
    if args.vbench_subsample:
        probe = cv2.VideoCapture(video_path)
        interval = max(1, round(probe.get(cv2.CAP_PROP_FPS) / 8))
        probe.release()
    frames, bgr, kept, src_fps = read_frames(video_path, device, interval)
    thres, count_num = vbench_threshold(frames[0], len(frames))
    n_pairs = len(frames) - 1 if args.max_pairs is None else min(args.max_pairs, len(frames) - 1)

    records = []
    with torch.no_grad():
        for i in range(n_pairs):
            image1, image2 = frames[i], frames[i + 1]
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up = padder.unpad(flow_up)  # dynamic_degree keeps padding; our videos need none, so identical
            max_rad, flow = get_score(flow_up)
            kind, label = classify_pair(kept[i], kept[i + 1], args.num_frame_per_block)
            records.append(dict(pair=i, f0=kept[i], f1=kept[i + 1], max_rad=max_rad, kind=kind, label=label, flow=flow))

    scale = args.flow_scale or max(r["max_rad"] for r in records) or 1.0

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_mp4 = os.path.join(args.out_dir, f"{stem}_flow.mp4")
    out_csv = os.path.join(args.out_dir, f"{stem}_flow.csv")
    h, w = records[0]["flow"].shape[:2]
    out_w = w * 2 if args.side_by_side else w
    # OpenCV can only write MPEG-4 part 2 ("mp4v"), which browsers / VS Code cannot play;
    # write that to a temp file and transcode to H.264 with ffmpeg afterwards.
    tmp_mp4 = out_mp4 + ".mp4v.tmp.mp4"
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps or src_fps / interval, (out_w, h))
    for r in records:
        vis = flow_to_bgr(r["flow"], scale)
        over = r["max_rad"] > thres
        lines = [
            (f"pair {r['pair']:3d}: frames {r['f0']} -> {r['f1']}", (255, 255, 255)),
            (f"max_rad = {r['max_rad']:.2f} px   ({'>' if over else '<='} thres {thres:.2f})", (255, 255, 255)),
            (r["label"], LABEL_COLOURS[r["kind"]]),
        ]
        vis = draw_overlay(vis, lines, r["kind"], wheel, scale)
        if args.side_by_side:
            src = bgr[r["pair"]].copy()
            cv2.putText(src, f"frame {r['f0']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            vis = np.hstack([src, vis])
        writer.write(vis)
    writer.release()
    to_h264(tmp_mp4, out_mp4)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair", "frame0", "frame1", "max_rad", "kind", "above_thres"])
        for r in records:
            wr.writerow([r["pair"], r["f0"], r["f1"], f"{r['max_rad']:.4f}", r["kind"], int(r["max_rad"] > thres)])

    # summary in the spirit of the lzh metrics
    by = {k: [r["max_rad"] for r in records if r["kind"] == k] for k in ("chunk", "latent", "none")}
    moving = sum(r["max_rad"] > thres for r in records)
    all_borders = by["chunk"] + by["latent"]
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    print(f"{stem[:70]}")
    print(f"   pairs={len(records)} interval={interval} thres={thres:.2f} pairs>thres={moving} (need {count_num}) "
          f"-> dynamic_degree={moving >= count_num}")
    print(f"   mean max_rad: chunk borders={mean(by['chunk']):.2f} (n={len(by['chunk'])})  "
          f"other latent borders={mean(by['latent']):.2f} (n={len(by['latent'])})  "
          f"intra-latent={mean(by['none']):.2f} (n={len(by['none'])})  "
          f"chunk/all-latent-borders={mean(by['chunk']) / mean(all_borders):.3f}  flow scale={scale:.2f}px")
    print(f"   -> {out_mp4}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_path", required=True, help="folder of .mp4 files, or a single .mp4")
    p.add_argument("--out_dir", default=None, help="default: <videos_path>/flow_viz")
    p.add_argument("--model", default=f"{CACHE_DIR}/raft_model/models/raft-things.pth",
                   help="RAFT checkpoint (default resolves like vbench.utils.init_submodules via VBENCH_CACHE_DIR)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_frame_per_block", type=int, default=3, help="latents per AR chunk (3 in all our configs)")
    p.add_argument("--vbench_subsample", action="store_true",
                   help="reproduce dynamic_degree.get_frames (resample to ~8 fps) instead of using every frame")
    p.add_argument("--flow_scale", type=float, default=None,
                   help="flow magnitude (px) mapped to full colour saturation; default = max max_rad in the video")
    p.add_argument("--fps", type=float, default=None, help="output fps (default: source fps / interval)")
    p.add_argument("--side_by_side", action="store_true", help="put the source frame left of the flow map")
    p.add_argument("--no_wheel", action="store_true", help="omit the colour-wheel legend")
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--max_pairs", type=int, default=None, help="debug: only the first N pairs of each video")
    args = p.parse_args()

    if os.path.isfile(args.videos_path):
        videos = [args.videos_path]
        base = os.path.dirname(args.videos_path)
    else:
        videos = sorted(glob.glob(os.path.join(args.videos_path, "*.mp4")))
        base = args.videos_path
    assert videos, f"no .mp4 found under {args.videos_path}"
    if args.max_videos:
        videos = videos[: args.max_videos]
    args.out_dir = args.out_dir or os.path.join(base, "flow_viz")
    os.makedirs(args.out_dir, exist_ok=True)
    assert os.path.isfile(args.model), f"RAFT checkpoint not found: {args.model} (set VBENCH_CACHE_DIR or --model)"

    device = torch.device(args.device)
    model = load_model(args.model, device)
    wheel = None if args.no_wheel else colour_wheel(72)
    print(f"RAFT: {args.model}\nvideos: {len(videos)}\nout: {args.out_dir}\n")
    for v in videos:
        process_video(v, model, device, args, wheel)


if __name__ == "__main__":
    main()
