import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

from vbench.utils import load_dimension_info

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


# pixel-frame pair index (frames i -> i+1 of the full-frame-rate video) that straddles the
# first chunk boundary: Wan latent k covers pixel frames 4k-3..4k, chunks are 3 latents,
# so chunk 0 = pixel frames 0..8 and the chunk 0|1 boundary is pair 8 (frames 8 -> 9).
FIRST_CHUNK_BORDER_PAIR = 8


class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}


    def infer(self, video_path):
        with torch.no_grad():
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
                frames_all = self.get_all_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
                raise ValueError(f"For AR project, dynamic degree metric does not support image folder input: {video_path}")
            else:
                raise NotImplementedError
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []

            latent_frame_border_score = []
            chunk_border_score = []
            first_chunk_border_score = 0

            for i, (image1, image2) in enumerate(zip(frames[:-1], frames[1:])):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)

            # mean diff-of-diff metrics: how much the frame-to-frame RGB change itself changes
            # between consecutive pairs (second temporal difference of the pixel values).
            #   d_i    = frame[i+1] - frame[i]                 (signed, per pixel & channel, 0..255 scale)
            #   adod_i = |d_i - d_{i-1}|                       (abs diff of signed diffs)
            #   adoad_i = | |d_i| - |d_{i-1}| |                (abs diff of abs diffs)
            # per video: m_*   = mean over pairs i>0 of the per-pair mean
            #            mm5_* = mean over pairs i>0 of the per-pair mean of the top-5% values
            #            mm5_*_c1 = the per-pair top-5% value at the first chunk boundary only:
            #                       d_{i-1} = frames 7->8 (last pair inside chunk 0), d_i = frames 8->9
            #                       (the chunk 0|1 boundary), i.e. loop index i == FIRST_CHUNK_BORDER_PAIR
            #            mm5_l2diff_c1 = mm5(||d_8||_2) - mm5(||d_7||_2): per pixel, ||d_i||_2 is the L2 norm of
            #                       the signed frame difference over the 3 colour channels (RGB Euclidean distance,
            #                       [1,H,W]); mm5(.) is the mean of its top-5% values. Positive = the first chunk
            #                       boundary (frames 8->9) changes more than the pair just before it (frames 7->8).
            prev_d, prev_ad, prev_l2_mm5 = None, None, None
            m_adod_list, m_adoad_list, mm5_adod_list, mm5_adoad_list = [], [], [], []
            mm5_adod_c1, mm5_adoad_c1, mm5_l2diff_c1 = float('nan'), float('nan'), float('nan')

            for i, (image1, image2) in enumerate(zip(frames_all[:-1], frames_all[1:])):
                d = image2 - image1          # unpadded tensors, [1,3,H,W]
                ad = d.abs()
                l2_mm5 = self.top5_mean(d.pow(2).sum(dim=1).sqrt())   # top-5% mean of per-pixel RGB L2 distance
                if i == FIRST_CHUNK_BORDER_PAIR and prev_l2_mm5 is not None:
                    mm5_l2diff_c1 = l2_mm5 - prev_l2_mm5
                if prev_d is not None:
                    adod = (d - prev_d).abs()
                    adoad = (ad - prev_ad).abs()
                    m_adod_list.append(adod.mean().item())
                    m_adoad_list.append(adoad.mean().item())
                    mm5_adod_list.append(self.top5_mean(adod))
                    mm5_adoad_list.append(self.top5_mean(adoad))
                    if i == FIRST_CHUNK_BORDER_PAIR:
                        mm5_adod_c1, mm5_adoad_c1 = mm5_adod_list[-1], mm5_adoad_list[-1]
                prev_d, prev_ad, prev_l2_mm5 = d, ad, l2_mm5

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)

                if i == FIRST_CHUNK_BORDER_PAIR:
                    first_chunk_border_score = max_rad

                if i%12 == 8:
                    chunk_border_score.append(max_rad)

                if i%4 == 0:
                    latent_frame_border_score.append(max_rad)

            whether_move = self.check_move(static_score)

            # LZH metrics (V4)
            chunk_border_score = np.mean(chunk_border_score)
            latent_frame_border_score = np.mean(latent_frame_border_score)

            chunk_inconsistency = chunk_border_score / (latent_frame_border_score + 1e-6)
            first_chunk_inconsistency = first_chunk_border_score / (latent_frame_border_score + 1e-6)

            dod_metrics = {
                'm_adod': float(np.mean(m_adod_list)),
                'm_adoad': float(np.mean(m_adoad_list)),
                'mm5_adod': float(np.mean(mm5_adod_list)),
                'mm5_adoad': float(np.mean(mm5_adoad_list)),
                'mm5_adod_c1': mm5_adod_c1,
                'mm5_adoad_c1': mm5_adoad_c1,
                'mm5_l2diff_c1': mm5_l2diff_c1,
            }

            return whether_move, chunk_inconsistency, first_chunk_inconsistency, dod_metrics


    @staticmethod
    def top5_mean(x):
        """Mean of the largest 5% of the elements of tensor x (same statistic as get_score's max_rad)."""
        flat = x.flatten()
        k = max(1, int(flat.numel() * 0.05))
        return torch.topk(flat, k).values.mean().item()


    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(1, round(fps / 8))
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 

    def get_all_frames(self, video_path):
        # every decoded pixel frame, no fps-based subsampling; used for the
        # chunk / latent-frame border scores so that pair i == pixel frames (i, i+1)
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list



AGGREGATE_METRIC_NAMES = ['dynamic_degree', 'm_adod', 'm_adoad', 'mm5_adod', 'mm5_adoad', 'mm5_adod_c1', 'mm5_adoad_c1', 'mm5_l2diff_c1']


def aggregate_video_results(video_results):
    """Per-dataset values for AGGREGATE_METRIC_NAMES: dynamic_degree is the fraction of
    moving videos ('video_results'), the rest are means of the per-video values."""
    values = [float(np.mean([d['video_results'] for d in video_results]))]
    for name in AGGREGATE_METRIC_NAMES[1:]:
        values.append(float(np.mean([d[name] for d in video_results])))
    return values


def dynamic_degree(dynamic, video_list):
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        score_per_video, chunk_inconsistency, first_chunk_inconsistency, dod_metrics = dynamic.infer(video_path)
        video_results.append(
            {
                'video_path': video_path,
                'video_results': score_per_video,
                'chunk_inconsistency': chunk_inconsistency,
                'first_chunk_inconsistency': first_chunk_inconsistency,
                **dod_metrics,
            }
            )
    return aggregate_video_results(video_results), video_results



def compute_dynamic_degree(json_dir, device, submodules_list, **kwargs):
    model_path = submodules_list["model"] 
    # set_args
    args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    dynamic = DynamicDegree(args_new, device)
    video_list, _ = load_dimension_info(json_dir, dimension='dynamic_degree', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = dynamic_degree(dynamic, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = aggregate_video_results(video_results)
    # json layout: [metric names, metric values, per-video dicts]
    # (upstream VBench: [dynamic_degree value, per-video dicts])
    return [list(AGGREGATE_METRIC_NAMES), all_results, video_results]
