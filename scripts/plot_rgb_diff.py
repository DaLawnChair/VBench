#!/usr/bin/env python3
"""Visualise per-pixel RGB change between consecutive frames of a video.

For every input video with n frames this writes an (n-1)-frame H.264 video.
Frame i shows the source frame i on the left and, on the right, a grayscale
map of the Euclidean distance between the RGB values of frames i and i+1:
    d(x, y) = sqrt((R1-R0)^2 + (G1-G0)^2 + (B1-B0)^2)
Black = no change, white = distance 255 (anything larger is clipped to 255).

Each frame carries a 50%-opacity box with the pair index, the mean distance
over the frame, the mean of the top-5% distances (the analogue of RAFT's
`max_rad` statistic in dynamic_degree), and whether the pair straddles a
chunk border, a latent-frame border, or neither.

Frame layout assumed (Wan2.1 VAE, 4x temporal compression):
  latent 0 -> pixel frame 0, latent k -> pixel frames 4k-3 .. 4k
  latent border k|k+1 lies between pixel frames 4k and 4k+1
  chunk border = latent border with (k+1) % num_frame_per_block == 0

CPU only; needs opencv + numpy (+ ffmpeg on PATH for the H.264 transcode).

Example:
  python3 scripts/plot_rgb_diff.py \\
      --videos_path /mnt/ddn/george/diffusion/john-AR/vbench_results/_stage1_double_cache_14b_msm_continued_10k/videos_rsample_seed1
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
def read_frames(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)  # BGR uint8; channel order does not matter for a Euclidean distance
    video.release()
    assert frames, f"no frames decoded from {video_path}"
    return frames, fps


def rgb_distance(f0, f1):
    d = f1.astype(np.float32) - f0.astype(np.float32)
    return np.sqrt(np.sum(d * d, axis=2))  # [H, W], range 0 .. 441.67


def top5_mean(x):
    flat = np.sort(x.ravel())[::-1]
    return float(flat[: max(1, int(flat.size * 0.05))].mean())


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


LABEL_COLOURS = {"chunk": (80, 80, 255), "latent": (60, 230, 255), "none": (230, 230, 230)}  # BGR
LABEL_THICKNESS = {"chunk": 2, "latent": 1, "none": 1}


def draw_overlay(img, lines, label_kind):
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
    return img


def to_h264(src, dst):
    """Transcode to H.264 / yuv420p (+faststart) so the file plays in VS Code, browsers, etc."""
    if shutil.which("ffmpeg") is None:
        print("   WARNING: ffmpeg not found, leaving MPEG-4 part 2 output (may not play in VS Code)")
        os.replace(src, dst)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)
    os.remove(src)


# --------------------------------------------------------------------------- #
def process_video(video_path, args):
    frames, src_fps = read_frames(video_path)
    n_pairs = len(frames) - 1 if args.max_pairs is None else min(args.max_pairs, len(frames) - 1)
    h, w = frames[0].shape[:2]

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_mp4 = os.path.join(args.out_dir, f"{stem}_rgbdiff.mp4")
    out_csv = os.path.join(args.out_dir, f"{stem}_rgbdiff.csv")
    tmp_mp4 = out_mp4 + ".mp4v.tmp.mp4"
    out_w = w if args.no_source else w * 2
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps or src_fps, (out_w, h))

    records = []
    for i in range(n_pairs):
        dist = rgb_distance(frames[i], frames[i + 1])
        mean_d, top5_d = float(dist.mean()), top5_mean(dist)
        kind, label = classify_pair(i, i + 1, args.num_frame_per_block)
        records.append(dict(pair=i, mean=mean_d, top5=top5_d, kind=kind))

        gray = np.clip(dist, 0, 255).astype(np.uint8)          # black = 0, white = 255 (clipped)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lines = [
            (f"pair {i:3d}: frames {i} -> {i + 1}", (255, 255, 255)),
            (f"mean dist = {mean_d:.2f}   top-5% mean = {top5_d:.1f}   (255 = white)", (255, 255, 255)),
            (label, LABEL_COLOURS[kind]),
        ]
        vis = draw_overlay(vis, lines, kind)
        if not args.no_source:
            src = frames[i].copy()
            cv2.putText(src, f"frame {i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            vis = np.hstack([src, vis])
        writer.write(vis)
    writer.release()
    to_h264(tmp_mp4, out_mp4)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair", "frame0", "frame1", "mean_dist", "top5_mean_dist", "kind"])
        for r in records:
            wr.writerow([r["pair"], r["pair"], r["pair"] + 1, f"{r['mean']:.4f}", f"{r['top5']:.4f}", r["kind"]])

    by = {k: [r["mean"] for r in records if r["kind"] == k] for k in ("chunk", "latent", "none")}
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    all_borders = by["chunk"] + by["latent"]
    print(f"{stem[:70]}")
    print(f"   pairs={len(records)}  mean RGB distance: chunk borders={mean(by['chunk']):.2f} (n={len(by['chunk'])})  "
          f"other latent borders={mean(by['latent']):.2f} (n={len(by['latent'])})  "
          f"intra-latent={mean(by['none']):.2f} (n={len(by['none'])})  "
          f"chunk/all-latent-borders={mean(by['chunk']) / mean(all_borders):.3f}")
    print(f"   -> {out_mp4}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_path", required=True, help="folder of .mp4 files, or a single .mp4")
    p.add_argument("--out_dir", default=None, help="default: <videos_path>/rgbdiff_viz")
    p.add_argument("--num_frame_per_block", type=int, default=3, help="latents per AR chunk (3 in all our configs)")
    p.add_argument("--fps", type=float, default=None, help="output fps (default: source fps)")
    p.add_argument("--no_source", action="store_true", help="omit the source frame on the left")
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
    args.out_dir = args.out_dir or os.path.join(base, "rgbdiff_viz")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"videos: {len(videos)}\nout: {args.out_dir}\n")
    for v in videos:
        process_video(v, args)


if __name__ == "__main__":
    main()
